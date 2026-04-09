# Deployment Requirements & Cost Analysis
## InsightFace buffalo_l Face Verification Microservice

**Report date:** April 2026  
**Service:** `insightface` — FastAPI + ONNX Runtime, port 8002  
**Model:** InsightFace buffalo_l (ArcFace w600k_r50 + RetinaFace), ~330 MB  
**Runtime:** `onnxruntime-gpu>=1.19.2` with automatic CUDAExecutionProvider / CPUExecutionProvider fallback  

> **Pricing note:** All figures are in USD/month at 24/7 continuous running unless otherwise stated. Azure and AWS prices reflect US East regions (East US / us-east-1). GCP prices reflect us-central1. Hetzner prices reflect EUR converted at ~1.09 USD/EUR as of April 2026. Hetzner prices include their April 2026 price adjustment (30–37% increase across cloud server tiers). Where exact post-adjustment CX-series pricing was not confirmed in official documentation, figures are estimated by applying the ~35% increase to previously published rates and are flagged accordingly.

---

## 1. Minimum System Requirements

These are the absolute floor values — the service will start and handle single sequential requests without crashing, but has no headroom.

### CPU
**Minimum: 2 vCPUs**

ONNX Runtime's `CPUExecutionProvider` uses its own internal thread pool for parallelising matrix operations within a single inference pass. By default it sets `intra_op_num_threads` to the number of logical cores available. On a 1-vCPU machine, this collapses to single-threaded execution and inference time on RetinaFace + ArcFace can spike to 15–25 seconds per request. Two vCPUs give ONNX Runtime at least one thread for the operator graph and one for the OS/uvicorn, reducing inference latency to the 4–8 second range and keeping the process from being starved during startup.

### RAM
**Minimum: 3 GB usable (4 GB VM)**

Memory breakdown at minimum:
- buffalo_l model weights loaded into ONNX Runtime session: ~330 MB on disk; expanded to ~500–700 MB in RAM after ONNX graph optimisation and session initialisation
- ONNX Runtime internal allocator and intermediate activation buffers during inference: ~400–600 MB peak (RetinaFace operates on a 640×640 BGR image; ArcFace produces a 512-dim embedding — neither is memory-intensive per se, but ONNX Runtime allocates workspace buffers at session prep time)
- Python interpreter + FastAPI + uvicorn + pillow-heif + OpenCV baseline: ~150–250 MB
- CUDA image layers (nvidia/cuda:12.1.1-cudnn8-runtime): the container base image alone is ~3.5 GB on disk but does not consume RAM at runtime beyond loaded shared libraries (~50–100 MB for cuDNN stubs even in CPU-only mode)
- Incoming image buffers: two images at 5 MB each, decoded to raw BGR arrays (a 12 MP photo at 4032×3024×3 = ~36 MB uncompressed); peak in-flight allocation during a single request: ~80–120 MB

Total peak: approximately 1.3–1.7 GB. A 2 GB RAM instance is technically achievable under light load but leaves no margin for OS overhead, caching, or concurrent request handling during a slow request. **4 GB VM is the practical minimum.**

### Storage
**Minimum: 20 GB**

- nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 base image: ~3.5 GB compressed, ~5 GB unpacked
- Python packages (onnxruntime-gpu, insightface, opencv-python-headless, pillow-heif, scipy, fastapi, uvicorn): ~1.5 GB installed
- buffalo_l model cache volume (`/root/.insightface`): ~330 MB downloaded at first run
- OS root filesystem + swap: ~2–3 GB
- Total: ~10–11 GB. A 20 GB volume provides a safe minimum with room for log accumulation and OS updates.

### GPU
**Optional at minimum spec.** The service automatically falls back to `CPUExecutionProvider` if no CUDA device is found. CPU-only operation is slower (2–8 seconds per request at minimum spec) but fully functional.

### Network
**Minimum: 10 Mbps ingress.** Each request uploads two images at up to 5 MB each = 10 MB. At 10 Mbps that takes ~8 seconds to upload, which is already the inference budget — adequate for a low-traffic internal service. Egress response payloads are JSON under 1 KB.

---

## 2. Recommended System Requirements

These are the specs for a production deployment with comfortable headroom for request queuing, OS noise, and moderate traffic spikes.

### CPU
**Recommended: 4 vCPUs**

ONNX Runtime's thread pool for `intra_op_parallelism` scales well up to 4 threads for ResNet-50 class models. Benchmarks on ArcFace-equivalent networks show that moving from 2 to 4 cores reduces inference latency by roughly 30–45%. Beyond 4 cores, returns diminish significantly for a model of this size and the cost increase is rarely justified at low-to-medium traffic. 4 vCPUs also leave headroom for uvicorn's event loop, OpenCV image decoding (which can be CPU-intensive for HEIF via pillow-heif), and EXIF-aware transposition using PIL.

### RAM
**Recommended: 8 GB**

Doubling the minimum RAM gives the OS page cache room to operate, prevents the Python allocator from thrashing the kernel for small allocations during concurrent requests, and handles edge cases like large HEIF inputs (some HEIF images can decode to substantially larger raw arrays than standard JPEG/PNG at equivalent file sizes). It also provides safe margin if a future version of the InsightFace model pack is slightly larger.

### Storage
**Recommended: 30 GB**

Adds buffer for Docker layer updates, log rotation, and future model cache growth without requiring a resize operation.

### GPU
**Not required at recommended spec.** See Section 3c and Section 5 for a full GPU cost-benefit analysis. For a low-to-medium traffic internal service, the CPU-only recommended spec is the suggested starting point.

### Network
**Recommended: 100 Mbps ingress** for comfortable handling of concurrent uploads without upload-side queuing becoming the bottleneck.

---

## 3. Azure Cost Breakdown

Azure is the primary deployment target. All prices are Linux pay-as-you-go in **East US** unless noted.

### Additional shared costs (all options)

| Item | Monthly cost |
|---|---|
| Azure Managed Disk (P10, 128 GB Premium SSD) | ~$19.71 |
| Azure Container Registry (Basic tier) | $5.00 |
| Networking egress (estimated 10 GB/month at $0.087/GB after 5 GB free) | ~$0.44 |
| **Shared monthly overhead** | **~$25** |

The model cache is stored in a Docker volume backed by a managed disk. The service binary + base image fits comfortably in a 128 GB Premium SSD; a Standard HDD could drop this to ~$5/month but is not recommended in production due to IOPS sensitivity during container startup (model cache read at startup).

---

### 3a. CPU-Only — Minimum Spec

**VM: Standard_B2ms** — 2 vCPUs, 8 GB RAM

| Item | Cost |
|---|---|
| Standard_B2ms on-demand (East US) | **$60.74/month** |
| Shared infrastructure overhead | ~$25.00 |
| **Total** | **~$86/month** |

> The B-series is burstable. For a low-traffic service with mostly idle periods between requests, the B2ms is cost-effective — CPU credits accumulate during idle time and are spent during inference bursts. However, if the service receives sustained back-to-back requests (e.g., a batch of consent verifications), credits deplete and CPU is throttled to the 40% baseline. For a production microservice that must not degrade under load spikes, this is a risk worth noting.

**1-year reserved:** Standard_B2ms reserved ~$35–40/month (saving ~35%).

---

### 3b. CPU-Only — Recommended Spec

**VM: Standard_D4s_v5** — 4 vCPUs, 16 GB RAM (non-burstable, dedicated vCPUs)

| Item | Cost |
|---|---|
| Standard_D4s_v5 on-demand (East US) | ~**$140/month** |
| Shared infrastructure overhead | ~$25.00 |
| **Total** | **~$165/month** |

> The Dv5 series provides consistent, non-burstable compute, which is more appropriate for an inference service where latency predictability matters. At 4 vCPUs it gives ONNX Runtime a full parallel thread pool.

Alternatively, **Standard_B4ms** (4 vCPUs, 16 GB, burstable) is available at **$120.88/month** on-demand — a $44/month saving over the Dv5. It is acceptable if traffic is genuinely low-to-medium with long idle gaps between requests. If you expect burst patterns (multiple concurrent consent flows), prefer the Dv5.

**1-year reserved (B4ms):** ~$81.61/month, bringing total to ~$107/month.

---

### 3c. GPU Deployment — NVIDIA T4

**VM: Standard_NC4as_T4_v3** — 4 vCPUs, 28 GB RAM, 1× NVIDIA T4 (16 GB VRAM)

| Item | Cost |
|---|---|
| Standard_NC4as_T4_v3 on-demand (East US) | **$378.72/month** ($0.526/hr) |
| Shared infrastructure overhead | ~$25.00 |
| **Total** | **~$404/month** |

#### Is GPU worth it for this workload?

**Short answer: No, at this traffic level.**

The ArcFace ResNet-50 model processed via ONNX Runtime is a relatively small inference workload:

- The entire model is ~80–100 MB of ONNX weights. The T4's 16 GB VRAM is massively overprovisioned.
- The two ONNX passes (RetinaFace detection on 640×640 + ArcFace embedding) are sequential and single-batch. GPU acceleration is most beneficial for large batch inference or transformer-class models with heavy attention layers.
- Expected GPU inference time: under 1 second. Expected CPU inference time on 4 vCPUs: 2–4 seconds.

The latency improvement is real (roughly 3–5× faster) but the absolute numbers matter more than the ratio. For an internal verification service called by a NestJS backend — where a 2–4 second response is acceptable — shaving that to under 1 second does not change user experience in any meaningful way. The NestJS caller will already have uploaded the images (which takes 1–3 seconds at typical connection speeds) before inference begins.

**Cost comparison:**
- Recommended CPU spec (B4ms, 1-year reserved): ~$107/month
- GPU spec (NC4as_T4_v3, on-demand): ~$404/month
- GPU spec (NC4as_T4_v3, spot): ~$0.194/hr → **~$166/month** (but spot is interruptible — incompatible with the always-on requirement)

The GPU configuration costs 3.8× more at on-demand pricing for a latency improvement that does not improve the end-user experience of the product. **GPU is not recommended for this service in isolation.** The exception would be if this service is co-deployed on infrastructure that already has GPU capacity provisioned for another workload (e.g., the Florence-2 OCR service on the same host), in which case the marginal GPU cost is already paid.

---

## 4. Alternative Services Cost Breakdown

### 4a. AWS

**Minimum spec equivalent: t3.large** — 2 vCPUs, 8 GB RAM

| Item | Cost |
|---|---|
| t3.large on-demand (us-east-1) | **$60.74/month** |
| EBS gp3 30 GB volume | ~$2.40 |
| ECR (Elastic Container Registry, 500 MB storage) | ~$0.05 |
| Egress (~10 GB) | ~$0.90 |
| **Total** | **~$64/month** |

**Recommended spec equivalent: t3.xlarge** — 4 vCPUs, 16 GB RAM

| Item | Cost |
|---|---|
| t3.xlarge on-demand (us-east-1) | **$121.47/month** |
| EBS gp3 30 GB + ECR + egress | ~$4 |
| **Total** | **~$125/month** |

**1-year Savings Plan (t3.xlarge):** ~$0.052/hr → **~$37.44/month**, total ~$42/month — a substantial saving over Azure's reserved pricing for equivalent spec.

**GPU equivalent: g4dn.xlarge** — 4 vCPUs, 16 GB RAM, 1× NVIDIA T4 (16 GB VRAM)

- On-demand: **$383.98/month**
- 1-year reserved: ~$230/month
- Spot: ~$0.2146/hr → **~$156/month** (interruptible)

**Notes:**
- AWS t3 instances are also burstable (same credit model as Azure B-series). For the recommended spec, consider **m6i.xlarge** (4 vCPUs, 16 GB, non-burstable) at ~$140/month on-demand, $84/month with 1-year Savings Plan — comparable to Azure Dv5.
- AWS has more mature ONNX Runtime tooling documentation and Deep Learning AMIs, which can reduce container setup friction.
- EBS volumes must be explicitly backed up; snapshot costs are additional (~$0.05/GB/month for stored snapshots).

---

### 4b. Google Cloud Platform (GCP)

**Deployment model: Cloud Run (instance-based billing) with minimum-instances=1**

Cloud Run with `--min-instances=1` keeps one container always warm, satisfying the always-on requirement. Billing is at the instance-based rate (charged continuously, not just during requests).

**Minimum spec equivalent: 2 vCPU, 4 GB RAM, min-instances=1**

At instance-based billing rates (us-central1):
- CPU: 2 vCPU × 2,678,400 seconds/month × $0.000024/vCPU-second = ~$128.56
- Memory: 4 GiB × 2,678,400 × $0.0000025 = ~$26.78
- **Total compute: ~$155/month**

> This is significantly more expensive than an equivalent VM for an always-on workload. Cloud Run's pricing model optimises for bursty/intermittent traffic — it penalises always-on services relative to VMs.

**Recommended spec equivalent: Compute Engine n2-standard-4** — 4 vCPUs, 16 GB RAM

| Item | Cost |
|---|---|
| n2-standard-4 on-demand (us-central1) | **$141.79/month** |
| Persistent disk 30 GB | ~$1.50 |
| Artifact Registry | ~$0.50 |
| Egress | ~$1.20 |
| **Total** | **~$145/month** |

With GCP's **sustained use discount** (automatic for instances running >25% of the month — which always-on qualifies for), this drops to approximately **$102/month** — no reservation needed.

**1-year CUD (Committed Use Discount):** ~37% off → ~$89/month, total ~$93/month.

**GPU equivalent: n1-standard-4 + 1× NVIDIA T4**

- On-demand: ~$0.95/hr including T4 → **~$693/month** (GCP GPU instances are expensive)
- 1-year CUD: ~$410/month

GCP GPU pricing is considerably higher than AWS or Azure for T4-class instances.

**Notes:**
- GCP's automatic sustained use discounts make it one of the most cost-effective platforms for genuinely always-on workloads without requiring upfront commitments.
- Cloud Run is not recommended for this service due to the always-on, memory-heavy model load pattern.
- Artifact Registry (GCP's container registry) is priced at $0.10/GB/month storage — very cheap for a ~3 GB image.

---

### 4c. RunPod (GPU-focused cloud)

RunPod is a community GPU marketplace and managed pod platform. It offers T4 and A10 instances via both on-demand pods and a spot-style "community cloud."

**Minimum spec equivalent: CPU Pod, 4 vCPU, 8 GB RAM**

RunPod's CPU pods are not its primary product and are less well-documented. Rough pricing: ~$15–25/month for 4 vCPU / 8 GB.

**GPU equivalent: NVIDIA T4 pod (community cloud)**

- T4 (16 GB VRAM): approximately **$0.40/hr** → **~$292/month** at 24/7 running
- Secure Cloud T4 (SOC2 compliant): approximately **$0.50–0.60/hr** → **~$365–438/month**

**A10 (24 GB VRAM):**
- Community cloud: approximately **$0.39–0.69/hr** depending on availability → **~$285–504/month**

**Advantage over Azure/AWS GPU:** RunPod bills per second and has no minimum commitment, making it practical to test GPU inference at low cost before committing to an always-on configuration. However, for a genuine always-on production service, the community cloud is not reliable — pods can be reclaimed. RunPod Secure Cloud pods are more stable but approach Azure spot pricing.

**Storage:** RunPod charges separately for network volumes ($0.07/GB/month). A 30 GB volume for the model cache costs ~$2.10/month.

**Key caveat:** RunPod is not enterprise-grade. It lacks SLAs, native VPC integration, and managed identity features that a production backend like `inrcliq-backend` may eventually require. It is best suited for experimentation or as a cost-bridge while evaluating whether GPU is necessary.

---

### 4d. Hetzner Cloud

Hetzner is a German cloud provider with data centres in Nuremberg, Falkenstein, Helsinki, Ashburn (Virginia), and Hillsboro (Oregon). It has no GPU cloud offering — this comparison is for CPU-only configurations.

> **Pricing note:** Hetzner increased all cloud server prices on 1 April 2026 by 30–37%. Prices below reflect the post-adjustment rates. Exact new CX-series prices from the official adjustment page were not fully enumerated at time of writing; figures marked *(est.)* are based on applying the ~35% adjustment to previously published rates.

**Minimum spec equivalent: CX32** — 4 vCPUs, 8 GB RAM, 80 GB NVMe SSD

- Pre-adjustment: €6.80/month → Post-adjustment: **~€9.20/month** *(est.)* → **~$10/month**

> Note: CX-series uses shared vCPUs. For an ONNX Runtime workload that periodically saturates CPU for 2–8 seconds per inference, the shared vCPU contention model is a real concern. Under noisy-neighbour conditions, inference latency can increase unpredictably. For this reason, the **CPX** (dedicated vCPU) series is preferred for any production inference workload.

**Recommended spec equivalent: CPX31** — 4 vCPUs (dedicated), 8 GB RAM, 160 GB NVMe SSD

- Pre-adjustment: ~€12.49/month → Post-adjustment: **~€16.90/month** *(est.)* → **~$18/month**

All CX and CPX plans include **20 TB of traffic/month** — effectively unlimited egress for this use case, and a significant advantage over AWS/Azure/GCP which charge per GB after a free tier.

| Item | Cost |
|---|---|
| CPX31 (4 vCPU dedicated, 8 GB) | ~$18/month *(est., post-adjustment)* |
| Additional volume for model cache (if needed) | ~$0.60/month (€0.056/GB/month × 10 GB) |
| Container Registry | Not included natively; use Docker Hub free tier or Hetzner Registry Add-on (~€1/month) |
| Egress | Included in 20 TB/month allowance |
| **Total** | **~$20/month** *(est.)* |

**Advantages:**
- By far the cheapest option for CPU-only always-on deployment after the April 2026 adjustment
- Included traffic allowance eliminates egress costs entirely for this use case
- NVMe SSD included in base price — no separate volume needed unless model cache must be on a separate disk
- Virginia and Oregon data centres suitable for US-based deployments

**Disadvantages:**
- No GPU offering — cannot upgrade to GPU without changing providers
- Smaller ecosystem: no managed container service, no native Kubernetes (must use third-party or self-hosted K3s)
- Less enterprise compliance coverage than hyperscalers (relevant for GDPR/data residency if storing biometric data, though this service is stateless)
- The April 2026 price increase of 30–37% was driven by hardware cost inflation and may not be the last adjustment

---

## 5. Recommendation

### Platform and tier to start with

**Start on Hetzner CPX31 (CPU-only) if cost is the primary constraint.** At an estimated ~$20/month all-in, it is 4–5× cheaper than equivalent Azure or AWS options. The four dedicated vCPUs are suitable for ONNX Runtime inference, and the 20 TB traffic allowance eliminates egress costs. The tradeoff is operational maturity (no managed container service, no SLA) and the inability to upgrade to GPU within the platform.

**Start on AWS t3.xlarge (CPU-only) if you want a mature platform with a clear GPU upgrade path.** At ~$125/month on-demand or ~$42/month with a 1-year Savings Plan, it sits in the middle of the cost range. If you later decide GPU is justified, switching to a g4dn.xlarge on the same account with the same ECS task definition is a single configuration change. AWS also has the most straightforward support for NVIDIA CUDA Docker images via its Deep Learning AMIs and ECR.

**Azure is the natural choice if `inrcliq-backend` is already running Azure resources** (Azure Container Apps, Azure Blob Storage, Azure Content Moderator — all mentioned in the platform architecture). The B4ms at ~$107/month (1-year reserved) is adequate for a low-to-medium traffic service, and co-locating the microservice in the same Azure region as the NestJS backend eliminates cross-provider egress costs. The Standard_D4s_v5 at ~$165/month is the more robust option if latency predictability is important.

### Whether GPU is worth it

**No, not for this service at this traffic level.**

ResNet-50 ArcFace is a fast, lightweight model. The improvement from CPU to GPU (2–4 seconds → under 1 second per request) is real but does not change the user-perceived outcome for an internal consent verification service. The cost delta is substantial: the cheapest always-on GPU option (Azure NC4as_T4_v3 on-demand at ~$404/month) costs approximately 4× the recommended CPU spec. Even at 1-year reserved rates, GPU costs roughly 2.5–3× more than CPU.

GPU becomes worth considering only if request throughput grows to the point where the CPU service is queuing requests for more than a few seconds — i.e., if multiple consent verifications are initiated concurrently. At the current service profile (internal NestJS backend calls, not direct user traffic), this threshold is unlikely to be reached.

### Handling the model cache volume

The buffalo_l model pack (~330 MB) is downloaded from the InsightFace model zoo on first container startup and written to `/root/.insightface`. This is a Docker named volume (`insightface_cache`) in the current compose configuration.

In production, treat the model cache as **infrastructure, not application data**:

1. **Preferred: Bake the model into the Docker image.** Pre-download the model weights at build time using a multi-stage Dockerfile and copy them into `/root/.insightface`. This eliminates first-run download latency (20–40 seconds), removes the dependency on the InsightFace model zoo being reachable at startup, and makes the model version deterministic. The image grows by ~330 MB (from ~5 GB to ~5.3 GB), which is an acceptable tradeoff.

2. **Alternative: Mount a persistent volume.** If baking the model into the image is not feasible (e.g., build environment cannot access the model zoo), use a managed persistent disk (Azure Managed Disk / AWS EBS / GCP Persistent Disk) mounted at startup. Ensure the volume is pre-populated before the first production deployment by running a model-pull container as a Kubernetes init container or a one-off ECS task.

3. **Avoid relying on external download at runtime.** The InsightFace model zoo is not a CDN with uptime guarantees. A first-run download failure will cause the container to start in a broken state and require manual intervention.

### Cost optimisation strategies for ONNX Runtime inference services

**Thread pool tuning.** By default, ONNX Runtime sets `intra_op_num_threads` to the machine's logical CPU count. On a 4-vCPU instance, this means 4 threads for operator-level parallelism. You can tune this at session creation:

```python
opts = ort.SessionOptions()
opts.intra_op_num_threads = 4   # match vCPU count
opts.inter_op_num_threads = 1   # single worker, as the service uses workers=1
```

This prevents ONNX Runtime from over-subscribing threads beyond what uvicorn's single worker can efficiently schedule.

**Graph optimisations.** Enable ONNX Runtime's built-in graph optimisation at the highest level:

```python
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
```

For ResNet-50 class models, this can reduce inference time by 10–20% on CPU with no accuracy impact.

**1-year reserved instances vs. on-demand.** For a service that must always be running, a 1-year reservation is the single most impactful cost reduction available. The savings are consistent across all providers:

| Provider | On-demand | 1-year reserved | Saving |
|---|---|---|---|
| Azure B4ms | $120.88 | ~$81.61 | ~33% |
| AWS t3.xlarge | $121.47 | ~$37.44 (Savings Plan) | ~69% |
| GCP n2-standard-4 | $141.79 | ~$89.33 (CUD) | ~37% |

AWS Savings Plans offer by far the steepest discount (up to 69% for 1-year compute commitments) and are the best-value commitment product across the three hyperscalers for this workload pattern.

**Spot/preemptible instances.** Spot instances are not compatible with the always-on requirement. Do not use spot for this service unless you implement a warm standby with automatic failover, which adds operational complexity disproportionate to the cost saving.

**Container registry costs.** All three hyperscalers charge for container registry storage. The InsightFace image is large (~5 GB with base image). Use a single private registry (Azure Container Registry Basic at $5/month, AWS ECR at ~$0.50/month for 5 GB, GCP Artifact Registry at $0.50/month) and tag images precisely — avoid pushing new tags for every CI build unless using a cleanup policy.

---

## Summary Cost Table

| Configuration | Platform | Instance | vCPUs | RAM | $/month (on-demand) | $/month (reserved/optimised) |
|---|---|---|---|---|---|---|
| CPU minimum | Azure | B2ms | 2 | 8 GB | ~$86 | ~$61 (1-yr) |
| CPU minimum | AWS | t3.large | 2 | 8 GB | ~$64 | ~$40 (1-yr SP) |
| CPU minimum | GCP | n2-standard-2 | 2 | 8 GB | ~$75* | ~$47 (CUD) |
| CPU minimum | Hetzner | CPX21 (est.) | 3 | 8 GB | ~$14 *(est.)* | N/A |
| CPU recommended | Azure | B4ms | 4 | 16 GB | ~$146 | ~$107 (1-yr) |
| CPU recommended | AWS | t3.xlarge | 4 | 16 GB | ~$125 | ~$42 (1-yr SP) |
| CPU recommended | GCP | n2-standard-4 | 4 | 16 GB | ~$145* | ~$93 (CUD) |
| CPU recommended | Hetzner | CPX31 (est.) | 4 | 8 GB | ~$20 *(est.)* | N/A |
| GPU (T4) | Azure | NC4as_T4_v3 | 4 | 28 GB | ~$404 | ~$250 (1-yr, est.) |
| GPU (T4) | AWS | g4dn.xlarge | 4 | 16 GB | ~$384 | ~$230 (1-yr) |
| GPU (T4) | RunPod | T4 Secure Pod | 4 | — | ~$365 | N/A |

*GCP n2-standard prices include automatic sustained use discount (~20%) for always-on instances.

> All figures should be verified against provider pricing calculators before commitment. Cloud pricing is subject to change and regional variation.

---

*Report generated April 2026. Pricing data sourced from Vantage Instances, Holori, CloudPrice.net, and official provider pricing pages. Hetzner CX/CPX post-April-2026-adjustment figures are estimates based on published 30–37% increase applied to prior rates. Confirm exact figures at hetzner.com/cloud before ordering.*
