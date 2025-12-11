import os
import sys
import builtins
import json
import glob
import time
import queue
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

import numpy as np
import pysam
import pod5
from Bio.SeqIO.QualityIO import FastqGeneralIterator

# ---- 更稳的打印（实时刷新） ----
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)
ROOT_STDOUT = sys.stdout
_original_print = builtins.print

def print(*args, **kwargs):
    if "file" not in kwargs:
        kwargs["file"] = ROOT_STDOUT
    if "flush" not in kwargs:
        kwargs["flush"] = True
    return _original_print(*args, **kwargs)


def _posix_fallocate_or_truncate(fd: int, size: int):
    try:
        os.posix_fallocate(fd, 0, size)
    except AttributeError:
        os.ftruncate(fd, size)
    except OSError:
        os.ftruncate(fd, size)


class BatchSignalExtractor:
    """按预规划写入顺序生成信号/索引/FASTQ/参考 FASTA."""

    FLUSH_THRESHOLD = 16 * 1024 * 1024  # bytes

    def __init__(self, bam_file, pod5_file, regions_data, output_base_dir,
                 num_threads=None, use_mv=False, flat_output=False,
                 num_consumer_threads=4, chr_writer_threads=1,
                 enable_queue_monitor=False):
        self.bam_file = bam_file
        self.pod5_file = pod5_file
        self.regions_data = regions_data
        self.output_base_dir = output_base_dir
        self.num_threads = num_threads or min(cpu_count(), 16)
        self.use_mv = use_mv
        self.flat_output = flat_output
        self.num_consumer_threads = max(1, int(num_consumer_threads or 4))
        self.writer_thread_count = max(1, int(chr_writer_threads or 1))

        # FASTQ / truncation
        self.all_fastq_reads = {}
        self.all_fastq_read_order = {}
        self.all_truncation_info = {}
        self.fastq_records = {}
        self.prebuilt_fastq_blocks = defaultdict(dict)
        self.region_ref_info = {}

        # mv / stride / idx1
        self.read_strides = {}
        self.read_moves = {}
        self.read_mv_idx1 = {}
        self.read_to_regions = defaultdict(set)

        # 规划结果
        self.chr_region_order = defaultdict(list)
        self.plan_entries_by_chr = defaultdict(list)
        self.plan_lookup = {}
        self.region_blocks = defaultdict(dict)
        self.chr_totals = {}
        self.pending_entries = set()
        self.pending_lock = threading.Lock()

        # 输出资源
        self.chr_resources = {}
        self.write_queue = queue.Queue(maxsize=8192)
        self.writer_threads = []
        self.queue_monitors = []
        self.enable_queue_monitor = bool(enable_queue_monitor)

        # POD5 阶段队列
        self.loaded_reads_queue = None
        self.loaded_read_ids = set()

        self.total_processed_files = 0
        self.total_loaded_count = 0
        self.completed_entries = 0

        print("mv:", use_mv)
        # 新增：按染色体聚合的输入文件映射与可选的区域筛选
        self.chr_files = {}  # {chr: {"fastq": path, "tsv": path}}
        self.allowed_regions_by_chr = None

    # ------------------------------------------------------------------
    # Stage 1/2/3 - 读取元数据
    # ------------------------------------------------------------------
    def load_all_fastq_reads(self):
        print("Step 1/5: Loading FASTQ reads...")
        if not self.chr_files:
            raise RuntimeError("chr_files is empty; please provide chromosome-level fastq/tsv inputs")

        # reset state
        self.all_fastq_reads.clear()
        self.all_fastq_read_order.clear()
        self.all_truncation_info.clear()
        self.prebuilt_fastq_blocks.clear()
        self.region_ref_info.clear()
        self.chr_region_order.clear()

        total_reads = 0

        def _tsv_iter(tsv_path):
            with open(tsv_path, 'r') as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    if line.lower().startswith("region_name"):
                        continue
                    parts = line.split('\t')
                    if len(parts) < 5:
                        raise RuntimeError(f"Invalid TSV line in {tsv_path}: {line}")
                    region_name, read_id, t_start, t_end, flag = parts[:5]
                    yield region_name, read_id, int(t_start), int(t_end), int(flag)

        for chr_name in sorted(self.chr_files.keys()):
            paths = self.chr_files[chr_name]
            fastq_path = paths.get("fastq")
            tsv_path = paths.get("tsv")
            if not fastq_path or not tsv_path or (not os.path.exists(fastq_path)) or (not os.path.exists(tsv_path)):
                raise RuntimeError(f"Missing FASTQ/TSV for {chr_name}")

            allowed = None
            if self.allowed_regions_by_chr and chr_name in self.allowed_regions_by_chr:
                allowed = self.allowed_regions_by_chr[chr_name]

            tsv_gen = _tsv_iter(tsv_path)
            with open(fastq_path, 'r') as fq:
                for title, seq, qual in FastqGeneralIterator(fq):
                    try:
                        region_name, read_id_tsv, t_start, t_end, flag = next(tsv_gen)
                    except StopIteration:
                        raise RuntimeError(f"TSV shorter than FASTQ for {chr_name}")

                    read_id = title.split()[0]
                    if read_id != read_id_tsv:
                        raise RuntimeError(f"FASTQ/TSV read_id mismatch in {chr_name}: {read_id} vs {read_id_tsv}")

                    if qual is None or len(qual) == 0:
                        qual = 'I' * len(seq)

                    if allowed is not None and region_name not in allowed:
                        continue

                    if region_name not in self.chr_region_order[chr_name]:
                        self.chr_region_order[chr_name].append(region_name)

                    self.prebuilt_fastq_blocks.setdefault(region_name, {})
                    self.all_fastq_read_order.setdefault(region_name, [])
                    self.all_fastq_reads.setdefault(region_name, set())
                    self.all_truncation_info.setdefault(region_name, {})

                    if read_id.startswith("REFERENCE_"):
                        ref_bucket = self.region_ref_info.setdefault(region_name, {})
                        hap_label = None
                        if read_id.endswith("_hap1"):
                            hap_label = "hap1"
                        elif read_id.endswith("_hap2"):
                            hap_label = "hap2"
                        ref_bucket[hap_label or "ref"] = {"read_id": read_id, "sequence": seq}
                    else:
                        block = f"@{read_id}\n{seq}\n+\n{qual}\n"
                        self.prebuilt_fastq_blocks[region_name][read_id] = block
                        self.all_fastq_read_order[region_name].append(read_id)
                        self.all_fastq_reads[region_name].add(read_id)
                        total_reads += 1

                    self.all_truncation_info[region_name][read_id] = {
                        'truncated_start': int(t_start),
                        'truncated_end': int(t_end),
                        'flag': int(flag),
                    }

                # 检查 TSV 是否有多余行
                try:
                    leftover = next(tsv_gen)
                    raise RuntimeError(f"TSV longer than FASTQ for {chr_name}: extra entry {leftover}")
                except StopIteration:
                    pass

        print(f"  Total reads loaded (non-reference): {total_reads}")

    def load_all_truncation_info(self):
        print("Step 2/5: Loading truncation info...")
        if self.all_truncation_info:
            print("  Truncation info already loaded from TSV; skipping.")
            return True
        print("  No truncation info found; ensure load_all_fastq_reads populated data.")
        return True

    def extract_all_signal_ranges_from_bam(self):
        print("Step 3/5: Extracting mv/stride from BAM...")
        read_interest = set()
        for region_reads in self.all_fastq_reads.values():
            if hasattr(region_reads, "keys"):
                read_interest.update(region_reads.keys())
            else:
                read_interest.update(region_reads)

        if not os.path.exists(self.bam_file):
            raise FileNotFoundError(self.bam_file)

        processed = matched = 0
        with pysam.AlignmentFile(self.bam_file, "rb", check_sq=False) as bam:
            for read in bam:
                processed += 1
                rid = read.query_name
                if rid not in read_interest:
                    continue
                if not read.has_tag('mv'):
                    continue
                mv_raw = read.get_tag('mv')
                if not mv_raw or len(mv_raw) < 2:
                    continue
                stride = int(mv_raw[0])
                if stride <= 0:
                    continue
                mv = np.frombuffer(bytearray(mv_raw[1:]), dtype=np.uint8).copy()
                idx1 = np.flatnonzero(mv)
                if idx1.size == 0:
                    continue
                self.read_strides[rid] = stride
                self.read_moves[rid] = mv
                self.read_mv_idx1[rid] = idx1
                matched += 1
        print(f"  BAM processed: {processed}, matched reads with mv: {matched}")

        for region_name, reads in self.all_fastq_reads.items():
            for rid in reads:
                if rid in self.read_moves:
                    self.read_to_regions[rid].add(region_name)

    # ------------------------------------------------------------------
    # Stage 3.5 - 规划 Plan 并预分配输出
    # ------------------------------------------------------------------
    def build_plan(self):
        print("Stage A: Building plan...")
        self.plan_entries_by_chr.clear()
        self.plan_lookup.clear()
        self.region_blocks.clear()
        self.chr_totals.clear()

        plan_entries_total = 0
        for chr_name, regions in self._iter_chr_regions_ordered():
            chr_sig_offset = 0
            chr_mv_offset = 0
            chr_entries = []
            chr_sig_total = 0
            chr_mv_total = 0

            for region_name in regions:
                fastq_order = self.all_fastq_read_order.get(region_name, [])
                region_entries = []
                region_sig = 0
                region_mv = 0

                for read_id in fastq_order:
                    entry = self._make_plan_entry(chr_name, region_name, read_id)
                    if not entry:
                        continue
                    entry['sig_off'] = chr_sig_offset + region_sig
                    entry['mv_off'] = chr_mv_offset + region_mv
                    region_sig += entry['sig_len']
                    region_mv += entry['mv_len']
                    region_entries.append(entry)
                    key = (region_name, read_id)
                    self.plan_lookup[key] = entry

                if region_entries:
                    block = {
                        'sig_off': chr_sig_offset,
                        'sig_len': region_sig,
                        'mv_off': chr_mv_offset,
                        'mv_len': region_mv,
                        'count': len(region_entries),
                    }
                    self.region_blocks[chr_name][region_name] = block
                    chr_sig_offset += region_sig
                    chr_mv_offset += region_mv
                    chr_entries.extend(region_entries)
                    plan_entries_total += len(region_entries)

            self.plan_entries_by_chr[chr_name] = chr_entries
            self.chr_totals[chr_name] = {
                'sig_len': chr_sig_offset,
                'mv_len': chr_mv_offset,
                'count': len(chr_entries),
            }

        self.pending_entries = set(self.plan_lookup.keys())
        print(f"  Plan entries: {plan_entries_total}")
        if plan_entries_total == 0:
            raise RuntimeError("Plan is empty. Check inputs.")

    def _make_plan_entry(self, chr_name, region_name, read_id):
        mv = self.read_moves.get(read_id)
        idx1 = self.read_mv_idx1.get(read_id)
        stride = self.read_strides.get(read_id)
        if mv is None or idx1 is None or stride is None:
            return None
        B = int(idx1.size)
        if B == 0:
            return None

        trunc = self.all_truncation_info.get(region_name, {}).get(read_id, {})
        x = int(trunc.get('truncated_start', 0))
        y = int(trunc.get('truncated_end', B))
        flag = int(trunc.get('flag', 0))
        if x < 0:
            x = 0
        if y > B:
            y = B
        if y <= x:
            return None

        t_start = int(idx1[x])
        t_end = int(idx1[y - 1])
        mv_len = t_end - t_start + 1
        if mv_len <= 0:
            return None
        sig_len = mv_len * int(stride)
        if sig_len <= 0:
            return None

        return {
            'chr': chr_name,
            'region': region_name,
            'read_id': read_id,
            'start_base': x,
            'end_base': y,
            'stride': int(stride),
            'flag': flag,
            'mv_tick_start': t_start,
            'mv_len': mv_len,
            'sig_len': sig_len,
        }

    def prepare_outputs_from_plan(self):
        print("Stage B: Preparing outputs (preallocate + index/FASTQ)...")
        self.chr_resources.clear()

        for chr_name, entries in self.plan_entries_by_chr.items():
            chr_dir = os.path.join(self.output_base_dir, chr_name)
            os.makedirs(chr_dir, exist_ok=True)

            totals = self.chr_totals[chr_name]
            sig_len = totals['sig_len']
            mv_len = totals['mv_len']

            sig_tmp = os.path.join(chr_dir, "signals.f32.bin.tmp")
            mv_tmp = os.path.join(chr_dir, "mv.u1.bin.tmp")
            sig_final = os.path.join(chr_dir, "signals.f32.bin")
            mv_final = os.path.join(chr_dir, "mv.u1.bin")
            idx_tmp = os.path.join(chr_dir, "index.tmp.tsv")
            idx_final = os.path.join(chr_dir, "index.tsv")
            fq_tmp = os.path.join(chr_dir, "reads.tmp.fastq")
            fq_final = os.path.join(chr_dir, "reads.fastq")
            region_blocks_path = os.path.join(chr_dir, "region_blocks.tsv")

            # 预分配二进制文件
            fd_sig = os.open(sig_tmp, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
            fd_mv = os.open(mv_tmp, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
            try:
                _posix_fallocate_or_truncate(fd_sig, sig_len * 4)
                _posix_fallocate_or_truncate(fd_mv, mv_len)
            finally:
                os.close(fd_sig)
                os.close(fd_mv)

            sig_mem = np.memmap(sig_tmp, dtype=np.float32, mode='r+', shape=(sig_len,)) if sig_len else None
            mv_mem = np.memmap(mv_tmp, dtype=np.uint8, mode='r+', shape=(mv_len,)) if mv_len else None

            # index & fastq 预写
            with open(idx_tmp, 'w') as idx_f:
                idx_f.write('\t'.join([
                    'chr','region','read_id',
                    'sig_offset','sig_len','mv_offset','mv_len',
                    'start_base','end_base','stride','flag'
                ]) + '\n')
                for entry in entries:
                    idx_f.write('\t'.join(map(str, [
                        entry['chr'], entry['region'], entry['read_id'],
                        entry['sig_off'], entry['sig_len'],
                        entry['mv_off'], entry['mv_len'],
                        entry['start_base'], entry['end_base'],
                        entry['stride'], entry['flag'],
                    ])) + '\n')

            with open(fq_tmp, 'w') as fq_f:
                for entry in entries:
                    block = self._get_fastq_block(entry['region'], entry['read_id'])
                    if not block:
                        raise RuntimeError(f"Missing FASTQ for {entry['region']}:{entry['read_id']}")
                    fq_f.write(block)

            with open(region_blocks_path, 'w') as rb:
                rb.write('\t'.join(['chr','region','sig_off','sig_len','mv_off','mv_len','count']) + '\n')
                for region_name, block in self.region_blocks[chr_name].items():
                    rb.write('\t'.join(map(str, [
                        chr_name, region_name,
                        block['sig_off'], block['sig_len'],
                        block['mv_off'], block['mv_len'],
                        block['count'],
                    ])) + '\n')

            self.chr_resources[chr_name] = {
                'dir': chr_dir,
                'sig_tmp': sig_tmp,
                'mv_tmp': mv_tmp,
                'sig_final': sig_final,
                'mv_final': mv_final,
                'idx_tmp': idx_tmp,
                'idx_final': idx_final,
                'fq_tmp': fq_tmp,
                'fq_final': fq_final,
                'sig_mem': sig_mem,
                'mv_mem': mv_mem,
                'sig_bytes_since_flush': 0,
                'mv_bytes_since_flush': 0,
            }
        self._write_ref_fastas()

    def _get_fastq_block(self, region_name, read_id):
        if self.fastq_records:
            rec = self.fastq_records.get(read_id)
            if rec:
                h, s, p, q = rec
                return ''.join([h, s, p, q])
        return self.prebuilt_fastq_blocks.get(region_name, {}).get(read_id)

    def _write_ref_fastas(self):
        refs_by_chr = defaultdict(list)
        # 按染色体、区域顺序写出，参考可能有 hap1/hap2
        for chr_name, regions in self._iter_chr_regions_ordered():
            for region_name in regions:
                info = self.region_ref_info.get(region_name)
                if not info:
                    continue
                for key in ("hap1", "hap2"):
                    if key in info:
                        refs_by_chr[chr_name].append((info[key]['read_id'], info[key]['sequence']))
                for key, ref in info.items():
                    if key in ("hap1", "hap2"):
                        continue
                    refs_by_chr[chr_name].append((ref['read_id'], ref['sequence']))

        for chr_name, entries in refs_by_chr.items():
            chr_dir = os.path.join(self.output_base_dir, chr_name)
            os.makedirs(chr_dir, exist_ok=True)
            path = os.path.join(chr_dir, "refs.fasta")
            with open(path, 'w') as fh:
                for ref_id, seq in entries:
                    fh.write(f">{ref_id}\n")
                    for i in range(0, len(seq), 60):
                        fh.write(seq[i:i+60] + "\n")

    # ------------------------------------------------------------------
    # Stage 4 - POD5 & 定位写
    # ------------------------------------------------------------------
    def load_all_pod5_signals(self):
        print("Step 4/5: Loading POD5 signals & writing data...")
        needed_reads = {rid for (_, rid) in self.pending_entries}
        if not needed_reads:
            print("  No reads require POD5 signals. Nothing to do.")
            return

        self._start_writer_threads()
        if self.enable_queue_monitor:
            self._start_queue_monitor(self.write_queue, "WriteQueue")
        self.loaded_reads_queue = queue.Queue(maxsize=512)
        if self.enable_queue_monitor:
            self._start_queue_monitor(self.loaded_reads_queue, "LoadedReadsQueue")
        producer_done = threading.Event()

        def consumer_worker(cid):
            while True:
                try:
                    item = self.loaded_reads_queue.get(timeout=0.5)
                except queue.Empty:
                    if producer_done.is_set():
                        break
                    continue
                if item is None:
                    self.loaded_reads_queue.task_done()
                    break
                read_id, signal_data = item
                try:
                    self._handle_loaded_read(read_id, signal_data)
                except Exception as exc:
                    print(f"  [Consumer-{cid}] error: {exc}")
                    raise
                finally:
                    self.loaded_reads_queue.task_done()

        consumers = []
        for cid in range(self.num_consumer_threads):
            t = threading.Thread(target=consumer_worker, args=(cid,), daemon=True)
            t.start()
            consumers.append(t)

        pod5_files = self._collect_pod5_files()
        if not pod5_files:
            raise RuntimeError("No POD5 files found")
        print(f"  Found {len(pod5_files)} POD5 files")

        def producer_task(file_batch, tid):
            loaded = processed = 0
            for pod5_file in file_batch:
                try:
                    with pod5.Reader(pod5_file) as reader:
                        for read_record in reader:
                            rid = str(read_record.read_id)
                            if rid not in needed_reads:
                                continue
                            signal = np.asarray(read_record.signal_pa, dtype=np.float32)
                            if signal.size == 0:
                                continue
                            self.loaded_reads_queue.put((rid, signal))
                            loaded += 1
                except Exception as exc:
                    print(f"    [POD5-{tid}] failed on {os.path.basename(pod5_file)}: {exc}")
                processed += 1
            print(f"    [POD5-{tid}] processed {processed} files, loaded {loaded} reads")

        optimal_threads = min(self.num_threads, len(pod5_files), 32)
        batches = self._split_batches(pod5_files, optimal_threads)
        with ThreadPoolExecutor(max_workers=len(batches)) as executor:
            futures = [executor.submit(producer_task, batch, tid) for tid, batch in enumerate(batches)]
            for fut in futures:
                try:
                    fut.result()
                except Exception as exc:
                    print(f"  Producer raised: {exc}")
        producer_done.set()

        # 等消费者完成
        for t in consumers:
            t.join()

        # 等写入完成
        self.write_queue.join()
        for _ in self.writer_threads:
            self.write_queue.put(None)
        for t in self.writer_threads:
            t.join()

        if self.enable_queue_monitor:
            self._stop_queue_monitors()
        self._finalize_outputs()
        remaining = len(self.pending_entries)
        if remaining:
            print(f"  Warning: {remaining} plan entries missing signals")
        else:
            print("  All plan entries written")

    def _handle_loaded_read(self, read_id, signal_data):
        if read_id in self.loaded_read_ids:
            return False
        self.loaded_read_ids.add(read_id)

        regions = self.read_to_regions.get(read_id)
        if not regions:
            return False

        mv = self.read_moves.get(read_id)
        stride = self.read_strides.get(read_id)
        if mv is None or stride is None:
            return False

        for region_name in regions:
            key = (region_name, read_id)
            entry = self.plan_lookup.get(key)
            if not entry:
                continue
            mv_start = entry['mv_tick_start']
            mv_end = mv_start + entry['mv_len']
            sig_start = mv_start * entry['stride']
            sig_end = sig_start + entry['sig_len']
            if sig_end > signal_data.size:
                raise RuntimeError(f"Signal shorter than expected for {region_name}:{read_id}")
            if mv_end > mv.size:
                raise RuntimeError(f"mv shorter than expected for {region_name}:{read_id}")
            seg = np.array(signal_data[sig_start:sig_end], dtype=np.float32, copy=True)
            mv_sub = np.array(mv[mv_start:mv_end], dtype=np.uint8, copy=True)
            if seg.size != entry['sig_len'] or mv_sub.size != entry['mv_len']:
                raise RuntimeError(f"Length mismatch for {region_name}:{read_id}")
            self.write_queue.put((entry, seg, mv_sub))
            self.total_loaded_count += 1
        return True

    def _start_writer_threads(self):
        if self.writer_threads:
            return
        thread_count = self.writer_thread_count
        for wid in range(thread_count):
            t = threading.Thread(target=self._writer_worker, args=(wid,), daemon=True)
            t.start()
            self.writer_threads.append(t)
        print(f"  Started {len(self.writer_threads)} writer thread(s)")

    def _writer_worker(self, wid):
        while True:
            item = self.write_queue.get()
            if item is None:
                self.write_queue.task_done()
                break
            entry, seg, mv_sub = item
            try:
                self._write_entry(entry, seg, mv_sub)
            except Exception as exc:
                print(f"  [Writer-{wid}] error for {entry['region']}:{entry['read_id']}: {exc}")
                raise
            finally:
                self.write_queue.task_done()

    def _write_entry(self, entry, seg, mv_sub):
        chr_name = entry['chr']
        resource = self.chr_resources[chr_name]
        sig_mem = resource['sig_mem']
        mv_mem = resource['mv_mem']
        if sig_mem is not None:
            sig_mem[entry['sig_off']:entry['sig_off'] + entry['sig_len']] = seg
            resource['sig_bytes_since_flush'] += entry['sig_len'] * 4
            if resource['sig_bytes_since_flush'] >= self.FLUSH_THRESHOLD:
                sig_mem.flush()
                resource['sig_bytes_since_flush'] = 0
        if mv_mem is not None:
            mv_mem[entry['mv_off']:entry['mv_off'] + entry['mv_len']] = mv_sub
            resource['mv_bytes_since_flush'] += entry['mv_len']
            if resource['mv_bytes_since_flush'] >= self.FLUSH_THRESHOLD:
                mv_mem.flush()
                resource['mv_bytes_since_flush'] = 0
        with self.pending_lock:
            self.pending_entries.discard((entry['region'], entry['read_id']))
            self.completed_entries += 1

    def _finalize_outputs(self):
        print("Stage D: Finalizing outputs...")
        for chr_name, resource in self.chr_resources.items():
            sig_mem = resource['sig_mem']
            mv_mem = resource['mv_mem']
            if sig_mem is not None:
                sig_mem.flush()
                del sig_mem
            if mv_mem is not None:
                mv_mem.flush()
                del mv_mem
            for tmp_path, final_path in [
                (resource['sig_tmp'], resource['sig_final']),
                (resource['mv_tmp'], resource['mv_final']),
                (resource['idx_tmp'], resource['idx_final']),
                (resource['fq_tmp'], resource['fq_final']),
            ]:
                os.replace(tmp_path, final_path)
        print("  Outputs ready")

    # ------------------------------------------------------------------
    def process_all_regions(self):
        done = self.completed_entries
        total = len(self.plan_lookup)
        print(f"Step 5/5: Summary - {done}/{total} plan entries written")

    # ------------------------------------------------------------------
    def run_batch_pipeline(self):
        start_time = time.time()
        steps = [
            ("Loading FASTQ reads", self.load_all_fastq_reads),
            ("Loading truncation info", self.load_all_truncation_info),
            ("Extracting signal ranges", self.extract_all_signal_ranges_from_bam),
            ("Building plan", self.build_plan),
            ("Preparing outputs", self.prepare_outputs_from_plan),
            ("Loading POD5 signals", self.load_all_pod5_signals),
            ("Processing regions", self.process_all_regions),
        ]
        try:
            for name, fn in steps:
                print(f"\n--- Starting: {name} ---")
                t0 = time.time()
                fn()
                print(f"--- Completed: {name} ({time.time() - t0:.2f}s) ---")
            print(f"\n=== Pipeline completed successfully in {time.time() - start_time:.2f}s ===")
            return True
        except Exception as exc:
            print(f"\n=== Pipeline failed: {exc} ===")
            import traceback
            traceback.print_exc()
            return False

    # ------------------------------------------------------------------
    def _collect_pod5_files(self):
        if os.path.isdir(self.pod5_file):
            pod5_files = []
            for root, _, files in os.walk(self.pod5_file):
                for file in files:
                    if file.endswith('.pod5'):
                        pod5_files.append(os.path.join(root, file))
            return sorted(pod5_files)
        return [self.pod5_file]

    def _split_batches(self, items, parts):
        if parts <= 0:
            return [items]
        size = max(1, len(items) // parts)
        batches = []
        for i in range(parts):
            start = i * size
            end = len(items) if i == parts - 1 else (i + 1) * size
            batch = items[start:end]
            if batch:
                batches.append(batch)
        return batches or [items]

    def _iter_chr_regions_ordered(self):
        for region_data in self.regions_data:
            chr_name = self._chr_from_region(region_data['region_name'])
            # ensure key exists
            self.chr_region_order.setdefault(chr_name, [])
        for chr_name in sorted(self.chr_region_order.keys()):
            yield chr_name, self.chr_region_order[chr_name]

    def _chr_from_region(self, region_name: str) -> str:
        normalized = region_name.replace(':', '_').replace('-', '_')
        return normalized.split('_')[0]

    def _start_queue_monitor(self, q, label):
        stop = threading.Event()

        def _monitor():
            while not stop.is_set():
                try:
                    size = q.qsize()
                    maxsize = q.maxsize if q.maxsize > 0 else "inf"
                    print(f"[{label}] queue size: {size}/{maxsize}")
                except Exception:
                    pass
                if stop.wait(2.0):
                    break

        t = threading.Thread(target=_monitor, name=f"{label}-Monitor", daemon=True)
        t.start()
        self.queue_monitors.append((t, stop))

    def _stop_queue_monitors(self):
        for t, stop in self.queue_monitors:
            stop.set()
            try:
                t.join(timeout=2.0)
            except Exception:
                pass
        self.queue_monitors.clear()


# -------------------- CLI --------------------
def main():
    args = sys.argv[1:]
    flat_output = False
    limit_per_chr = None
    offset_per_chr = 0
    positional_args = []
    consumer_threads = 4
    chr_writer_threads = 1
    enable_queue_monitor = False
    selected_chroms = None  # 可选：仅处理指定的染色体集合

    idx = 0
    while idx < len(args):
        arg = args[idx]
        if arg == "--flat-output":
            flat_output = True
            idx += 1
        elif arg.startswith("--chroms"):
            if arg == "--chroms":
                idx += 1
                selected_chroms = set(args[idx].split(","))
                idx += 1
            else:
                selected_chroms = set(arg.split("=", 1)[1].split(","))
                idx += 1
        elif arg.startswith("--limit"):
            if arg == "--limit":
                idx += 1
                limit_per_chr = int(args[idx])
                idx += 1
            else:
                limit_per_chr = int(arg.split("=", 1)[1])
                idx += 1
        elif arg.startswith("--offset"):
            if arg == "--offset":
                idx += 1
                offset_per_chr = int(args[idx])
                idx += 1
            else:
                offset_per_chr = int(arg.split("=", 1)[1])
                idx += 1
        elif arg.startswith("--consumer-threads"):
            if arg == "--consumer-threads":
                idx += 1
                consumer_threads = int(args[idx])
                idx += 1
            else:
                consumer_threads = int(arg.split("=", 1)[1])
                idx += 1
        elif arg.startswith("--chr-writer-threads"):
            if arg == "--chr-writer-threads":
                idx += 1
                chr_writer_threads = int(args[idx])
                idx += 1
            else:
                chr_writer_threads = int(arg.split("=", 1)[1])
                idx += 1
        elif arg.startswith("--enable-queue-monitor"):
            enable_queue_monitor = True
            idx += 1 if arg == "--enable-queue-monitor" else 1
        else:
            positional_args.append(arg)
            idx += 1

    if len(positional_args) < 2:
        print("Usage: python get_signal_opz.py <regions_dir> <output_dir> [num_threads] [mv_flag]")
        sys.exit(1)

    regions_dir = positional_args[0]
    output_dir = positional_args[1]
    num_threads = None
    mv_flag = False
    if len(positional_args) >= 3:
        num_threads = int(positional_args[2])
    if len(positional_args) >= 4:
        mv_flag = (positional_args[3] == "1")

    os.makedirs(output_dir, exist_ok=True)

    # 新格式：按染色体目录聚合
    chr_dirs = sorted(glob.glob(os.path.join(regions_dir, "chr*")))
    if selected_chroms:
        chr_dirs = [d for d in chr_dirs if os.path.basename(d) in selected_chroms]
    regions_data = []
    chr_files = {}
    allowed_regions_by_chr = {}

    def _read_regions_from_tsv(tsv_path):
        regions = []
        seen = set()
        with open(tsv_path, 'r') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if line.lower().startswith("region_name"):
                    continue
                parts = line.split('\t')
                if not parts:
                    continue
                region_name = parts[0]
                if region_name not in seen:
                    regions.append(region_name)
                    seen.add(region_name)
        return regions

    for chr_dir in chr_dirs:
        chr_name = os.path.basename(chr_dir)
        fastq_candidates = glob.glob(os.path.join(chr_dir, "*_reads.fastq"))
        tsv_candidates = glob.glob(os.path.join(chr_dir, "*_truncation.tsv"))
        if not fastq_candidates or not tsv_candidates:
            continue
        fastq_path = fastq_candidates[0]
        tsv_path = tsv_candidates[0]

        regions_in_chr = _read_regions_from_tsv(tsv_path)
        if not regions_in_chr:
            continue

        start_idx = min(offset_per_chr, len(regions_in_chr))
        candidate = regions_in_chr[start_idx:]
        chosen = candidate[:limit_per_chr] if limit_per_chr is not None else candidate
        if not chosen:
            continue

        chr_files[chr_name] = {"fastq": fastq_path, "tsv": tsv_path}
        allowed_regions_by_chr[chr_name] = set(chosen)
        for region_name in chosen:
            regions_data.append({
                'region_name': region_name,
                'chr_name': chr_name,
            })

    if not regions_data:
        print("No regions found in provided regions_dir.")
        sys.exit(1)
# /home/user/zhangbaomu/basecall/code/test/test_data/chr1.mv.bam

    extractor = BatchSignalExtractor(
        bam_file="/home/user/zhangbaomu/basecall/data/PAU_dorado_latest/dorado_sup_5.2.0.fixed.bam",
        pod5_file="/home/user/zhanganqi/xx/basecall/data/hg002/08_26_24_R1041_UL_GIAB_HG002_1A/20240826_1948_6D_PAU71968_4ac46f50/pod5/",
        regions_data=regions_data,
        output_base_dir=output_dir,
        num_threads=num_threads,
        use_mv=mv_flag,
        flat_output=flat_output,
        num_consumer_threads=consumer_threads,
        chr_writer_threads=chr_writer_threads,
        enable_queue_monitor=enable_queue_monitor,
    )
    extractor.chr_files = chr_files
    extractor.allowed_regions_by_chr = allowed_regions_by_chr
    extractor.run_batch_pipeline()


if __name__ == "__main__":
    main()
