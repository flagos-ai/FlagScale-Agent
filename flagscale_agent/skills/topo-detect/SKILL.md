---
name: topo-detect
description: Detect hardware topology — NVLink, NUMA, RDMA, disks on single node. For multi-node, also run NCCL tests to verify
  cross-node communication bandwidth. Generates a YAML report with parallelism recommendations.
keywords:
- topology
- topo
- hardware
- detect
- nvlink
- rdma
- infiniband
- numa
- pcie
- benchmark
- 拓扑
- 硬件检测
- 硬件拓扑
- 拓扑检测
- nccl
parameters:
- name: output_path
  description: Path to write YAML topology report
  default: ./topo_report.yaml
requires: []
suggests: []
constraints:
- id: no_install_packages
  description: Do NOT install any packages during topology detection — only use tools already present
  trigger:
    tools:
    - shell
    keywords:
    - pip install
    - conda install
    - apt install
    - yum install
  prompt: Check if the agent is trying to install packages during topology detection
  correction: Only use tools already present on the system. If a tool is unavailable, note it and move on.
- id: io_benchmark_confirm
  description: IO benchmark writes temporary files — must confirm with user before running
  trigger:
    tools:
    - shell
    keywords:
    - dd if=
    - fio
    - iozone
    - bonnie
  prompt: Check if an IO benchmark is about to run without user confirmation
  correction: Ask user for confirmation before running IO benchmarks that write temporary files.
- id: cleanup_temp_files
  description: Clean up temporary benchmark files after IO tests
  trigger:
    keywords:
    - dd
    - fio
    - testfile
    - benchmark
  prompt: Check if temporary benchmark files were created and need cleanup
  correction: Clean up any temporary files created during IO benchmarks.
context_injection:
  always:
  - Execution Rules
  - Output
  by_tool:
    shell:
    - Compute Detection
    - Communication Detection
    - Storage Detection
---
# Hardware Topology Detection (Single Node)

Detect compute, communication, and storage topology on the current node. Collect structured data, generate a YAML report, print a human-readable summary, and save a one-line finding to memory.

## Output

- YAML file at `{output_path}` — machine-readable topology report
- Terminal summary — concise human-readable overview
- Memory entry (key: `node_topology`, type: `finding`) — one-line summary for future sessions

## Execution Rules

- Run each detection step in order. If a command fails or is unavailable, note it as "unavailable" in the report and move on — do NOT stop.
- Do NOT install any packages. Only use tools already present on the system.
- IO benchmark (Step 4) writes temporary files. Ask the user for confirmation before running it. Clean up test files after.
- Parse all command outputs yourself. Do NOT ask the user to interpret raw output.
- After all steps, assemble the YAML report, print the summary, and write memory.

---

## Step 1: Compute Detection

### 1a. GPU inventory

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version,pci.bus_id,compute_cap --format=csv,noheader && echo "GPU_COUNT=$(nvidia-smi -L | wc -l)"
```

The `GPU_COUNT=` line gives the exact GPU count. Use that number — never count nvidia-smi output lines manually.

If `nvidia-smi` is not found, record `gpus: unavailable` and skip to Step 1d.

### 1b. GPU interconnect topology

```bash
nvidia-smi topo -m
```

Parse the matrix to determine:
- Interconnect type between each GPU pair: NV# (NVLink), SYS, PHB, PXB, PIX, NODE
- If ALL GPU pairs show NV# links, interconnect type is "NVSwitch" (full mesh)
- If only some pairs have NV#, it is "NVLink-P2P" (partial)
- If no NV# links, it is "PCIe-only"
- Extract CPU Affinity and NUMA Affinity columns for each GPU

Also extract NIC rows from this matrix — they show NIC-to-GPU PCIe distance (used in Step 2).

### 1c. NVLink bandwidth

```bash
nvidia-smi nvlink -s
```

For each GPU, count the number of active NVLink links and read bandwidth per link (GB/s).

### 1d. CPU topology

```bash
lscpu
```

Extract: model name, sockets, cores per socket, threads per core, total threads, NUMA node count.

### 1e. System memory

```bash
free -g
```

Extract: total memory (GB), available memory (GB).

### 1f. NCCL version

```bash
python3 -c "import torch; print(torch.cuda.nccl.version())" 2>/dev/null || echo "NCCL version unavailable"
```

If PyTorch is not available, try:
```bash
dpkg -l 2>/dev/null | grep -i nccl || rpm -qa 2>/dev/null | grep -i nccl || echo "NCCL package not found"
```

Record the NCCL version — it affects multi-node communication performance and feature support.

---

## Step 2: Communication Detection

### 2a. RDMA devices

```bash
ls /sys/class/infiniband/ 2>/dev/null
```

If the directory is empty or missing, record `rdma: unavailable` and skip to Step 3.

### 2b. NIC details

For each device found:

```bash
cat /sys/class/infiniband/<device>/ports/1/rate
cat /sys/class/infiniband/<device>/ports/1/state
cat /sys/class/infiniband/<device>/device/numa_node
```

Determine NIC type:
- Rate contains "HDR" / "EDR" / "FDR" / "QDR" / "NDR" → InfiniBand
- Rate contains "GigE" or device name suggests RoCE → RoCE
- Otherwise → unknown

### 2c. NIC-GPU affinity

From the `nvidia-smi topo -m` output (already collected in Step 1b), read the NIC rows.

For each NIC, find which GPUs have the closest PCIe distance (PXB or PIX = close, SYS = far). Record the closest GPU indices and the distance code.

### 2d. GPUDirect RDMA

```bash
lsmod 2>/dev/null | grep -E 'nv_peer_mem|nvidia_peermem'
```

If either module is loaded, GPUDirect RDMA is available.

Fallback: check if the module files exist:
```bash
find /lib/modules/$(uname -r) -name '*peer_mem*' 2>/dev/null
```

### 2e. NCCL configuration

```bash
env | grep NCCL || echo "(no NCCL env vars set)"
```

Record any NCCL-related environment variables currently set.

---

## Step 3: Storage Detection

### 3a. Block devices

```bash
lsblk -d -o NAME,SIZE,TYPE,ROTA,TRAN,MODEL 2>/dev/null
```

Classify each device:
- ROTA=0 + TRAN=nvme → NVMe SSD
- ROTA=0 + TRAN=sata/sas → SATA/SAS SSD
- ROTA=1 → HDD
- TYPE=loop → skip

### 3b. Mount points and filesystem

```bash
df -hT 2>/dev/null
```

Record: mount path, filesystem type, total size, used percentage. Skip tmpfs/devtmpfs/overlay unless they are notably large.

### 3c. Shared storage detection

```bash
mount | grep -iE 'type (nfs|lustre|gpfs|ceph|fuse\.ceph|beegfs|panfs)' 2>/dev/null
```

For each shared mount found, record type, mount point, and source.

Additional details if available:
- NFS: `nfsstat -m 2>/dev/null` or `mount | grep nfs` for server and options
- Lustre: `lfs df 2>/dev/null` for OST distribution
- GPFS: `mmlsfs all 2>/dev/null` for filesystem parameters

---

## Step 4: IO Benchmark (Optional)

**IMPORTANT: Ask the user for confirmation before running this step.** Explain that it will write a temporary 1GB file and take about 10-30 seconds.

If user declines, record `io_benchmark: skipped` and proceed to Step 5.

### 4a. Sequential write (dd)

```bash
TEST_DIR=$(df --output=target / | tail -1)
dd if=/dev/zero of=${TEST_DIR}/topo_bench_tmp bs=1M count=1024 oflag=direct 2>&1
rm -f ${TEST_DIR}/topo_bench_tmp
```

Parse the output for throughput (MB/s or GB/s).

### 4b. Random read (fio, if available)

```bash
which fio >/dev/null 2>&1
```

If fio is available:
```bash
fio --name=topo_randread --rw=randread --bs=4k --size=256M --numjobs=4 --runtime=10 --time_based --direct=1 --group_reporting --output-format=json --filename=/tmp/topo_fio_tmp 2>/dev/null
rm -f /tmp/topo_fio_tmp
```

Parse JSON output for IOPS and bandwidth.

If fio is not available, record `random_read: fio not installed, skipped`.

---

## Step 5: Generate Report

### 5a. Assemble YAML

Combine all collected data into a single YAML document with this structure:

```yaml
timestamp: "<ISO 8601>"
hostname: "<hostname>"

compute:
  gpus:
    count: <int>
    model: "<string>"
    memory_per_gpu_gb: <int>
    compute_capability: "<string>"
    driver_version: "<string>"
  interconnect:
    type: "<NVSwitch|NVLink-P2P|PCIe-only|unavailable>"
    links_per_gpu: <int>
    bandwidth_per_link_gbps: <float>
    total_bisection_bw_gbps: <float>
    topology_matrix: "<raw matrix text>"
  cpu:
    model: "<string>"
    sockets: <int>
    cores_per_socket: <int>
    threads_per_core: <int>
    total_threads: <int>
  numa:
    nodes: <int>
    gpu_affinity:
      numa0: [<gpu indices>]
      numa1: [<gpu indices>]
  memory:
    total_gb: <int>
    available_gb: <int>

communication:
  rdma:
    available: <bool>
    type: "<InfiniBand|RoCE|none>"
    gpudirect_rdma: <bool>
    devices:
      - name: "<string>"
        rate: "<string>"
        state: "<string>"
        numa_node: <int>
  nic_gpu_affinity:
    <nic_name>:
      closest_gpus: [<indices>]
      distance: "<PXB|PIX|SYS|...>"
  nccl_env: {<key>: "<value>"}

storage:
  block_devices:
    - name: "<string>"
      size: "<string>"
      type: "<NVMe|SSD|HDD>"
      rotational: <bool>
  mount_points:
    - path: "<string>"
      filesystem: "<string>"
      total: "<string>"
      used_pct: <int>
  shared_storage:
    - type: "<nfs|lustre|gpfs|ceph|beegfs>"
      mount: "<string>"
      source: "<string>"
  io_benchmark:
    sequential_write_mbps: <float|"skipped">
    random_read_iops: <float|"skipped"|"fio not installed">
    test_path: "<string>"

recommendations:
  - "<string>"
```

Write this to `{output_path}` using the write_file tool.

### 5b. Print terminal summary

Print a concise summary grouped by Compute / Communication / Storage / Recommendations. Use the format shown in the example below — adapt numbers to actual detected values:

```
=== Hardware Topology Report ===

Compute:
  GPUs: 8x <model> (<mem>GB each)
  Interconnect: <type>, <links>x NVLink per GPU (<bw> GB/s each)
  CPU: <sockets>x <model> (<threads> threads)
  Memory: <total> GB (<avail> GB available)

Communication:
  RDMA: <type>, <count>x <rate>
  GPUDirect RDMA: <available|unavailable>
  NIC-GPU affinity: <summary>

Storage:
  Local: <fs_type> <size> (<used>% used)
  Shared: <type> <mount> (<size>)
  IO: <seq_write> MB/s seq write, <rand_iops> random IOPS

Recommendations:
  - <recommendation 1>
  - <recommendation 2>
  ...

Report saved to: {output_path}
```

### 5c. Write memory

Write multiple memory entries so other skills (e.g. training-helper) can consume topology data:

1. **Compute topology** — key: `topo_compute`, type: `finding`
   Content format: `gpu_count=<N> model=<name> mem_gb=<M> interconnect=<NVSwitch|NVLink-P2P|PCIe> nvlink_bw_gbps=<B>`
   Example: `gpu_count=8 model=A800-SXM4-80GB mem_gb=80 interconnect=NVSwitch nvlink_bw_gbps=200`

2. **Communication topology** — key: `topo_comm`, type: `finding`
   Content format: `rdma=<yes|no> type=<IB|RoCE|none> nic_count=<N> nic_rate=<rate> gdrdma=<yes|no>`
   Example: `rdma=yes type=IB nic_count=4 nic_rate=200Gb/s gdrdma=yes`

3. **Storage topology** — key: `topo_storage`, type: `finding`
   Content format: `local=<type>:<size> shared=<type>:<mount>:<size> seq_write_mbps=<N>`
   Example: `local=NVMe:14TB shared=lustre:/data:100TB seq_write_mbps=1200`

4. **Summary** — key: `node_topology`, type: `finding`
   A single line combining the above, e.g.: "8x A800 80GB NVSwitch, 4x IB HDR 200Gb/s, GPUDirect RDMA, 14TB NVMe"

---

## Step 6: Recommendations

Generate recommendations based on detected topology. Use these rules as a framework — adapt to actual findings:

### Compute recommendations
- NVSwitch full mesh (all NV#) → "TP up to <gpu_count> is efficient within this node"
- NVLink-P2P partial → "TP groups should follow NVLink connectivity — check topology matrix for optimal grouping"
- PCIe-only → "TP across GPUs will be slow — prefer PP or DP for multi-GPU parallelism"
- High GPU memory (>= 80GB) → "Large model shards fit per GPU — may reduce TP/PP degree needed"

### Communication recommendations
- Multiple IB NICs → "Aggregate bandwidth: <count> x <rate> = <total> — sufficient for <assessment>"
- GPUDirect RDMA available → "Enable NCCL_NET_GDR_LEVEL=5 for lowest latency GPU-NIC transfers"
- NIC-GPU affinity detected → "Bind NCCL to NICs closest to each GPU group for optimal RDMA performance"
- No RDMA → "No RDMA detected — inter-node communication will use TCP, expect lower bandwidth"

### Storage recommendations
- NVMe local storage → "Fast local storage available for checkpoint staging"
- Shared storage detected → "Shared filesystem at <mount> — suitable for dataset and checkpoint storage"
- High IO bandwidth → "IO bandwidth sufficient for data loading at <rate>"
- Low IO bandwidth or HDD → "Consider caching datasets to local NVMe to avoid IO bottleneck"
- Disk usage > 80% → "Warning: <mount> is <pct>% full — ensure sufficient space for checkpoints"

### NCCL recommendations
- NCCL version < 2.18 → "Consider upgrading NCCL for better multi-node performance and bug fixes"
- NCCL version >= 2.18 → "NCCL version supports latest features including NVLink SHARP and tree reduction"

---

## Multi-Node Topology Verification

This skill primarily detects topology on a single node. For multi-node clusters, add a lightweight cross-node verification step:

### Run on each node

Run this skill on each node separately and compare reports to verify homogeneous hardware. If nodes differ (e.g., different GPU models or NIC counts), flag this to the user — heterogeneous clusters require careful parallelism planning.

### Cross-Node Communication Test (NCCL Test)

After single-node detection, run NCCL tests to verify cross-node communication bandwidth. This does NOT require switch-level access — it runs entirely in user space.

**Check if nccl-tests is available:**
```bash
which all_reduce_perf 2>/dev/null || ls /usr/local/bin/all_reduce_perf 2>/dev/null || echo "nccl-tests not installed"
```

If not installed, suggest the user build it:
```bash
# Build nccl-tests (requires MPI and NCCL)
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make MPI=1 MPI_HOME=/usr/local/mpi CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr/local/nccl
```

**Run all_reduce bandwidth test across nodes:**
```bash
# 2-node example (adjust hostfile and -np for actual cluster)
mpirun -np <total_gpus> --hostfile <hostfile> \
  -x NCCL_DEBUG=INFO -x NCCL_IB_DISABLE=0 \
  all_reduce_perf -b 8 -e 2G -f 2 -g 1
```

**Parse results:**
- Look at the `busbw` column for large message sizes (>=256MB)
- Expected: close to theoretical NIC bandwidth × NIC count (e.g., 4x 200Gb/s IB = ~100 GB/s bus bandwidth)
- If significantly lower, check: NIC-GPU affinity, GPUDirect RDMA, NCCL environment variables

**Simplified multi-node check (without nccl-tests):**

If nccl-tests is not available and the user cannot install it, use PyTorch's built-in NCCL:
```bash
# Run on 2 nodes (adjust MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK per node)
torchrun --nproc_per_node=<gpus_per_node> --nnodes=2 \
  --master_addr=<node0_ip> --master_port=29500 \
  -c "
import torch, torch.distributed as dist, time
dist.init_process_group('nccl')
rank = dist.get_rank()
t = torch.randn(256*1024*1024 // 4, device='cuda')  # 256MB
dist.barrier()
start = time.time()
for _ in range(10):
    dist.all_reduce(t)
torch.cuda.synchronize()
elapsed = time.time() - start
if rank == 0:
    bw = 10 * 256 / elapsed / 1024  # GB/s
    print(f'AllReduce 256MB x10: {elapsed:.2f}s, ~{bw:.1f} GB/s')
dist.destroy_process_group()
"
```

Record cross-node bandwidth in memory:
- key: `topo_cross_node`, type: `finding`
- Content: `nodes=<N> allreduce_bw_gbps=<B> nccl_version=<V>`

---

## Related Skills

- `train-env-setup` — install FlagScale environment on detected hardware
- `train-config` — use topology data to configure parallelism strategy
- `train-run` — launch training with topology-aware settings
