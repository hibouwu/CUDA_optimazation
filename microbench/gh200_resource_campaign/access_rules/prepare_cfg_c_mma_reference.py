"""Derive one isolated instruction-service probe without changing R00."""
from pathlib import Path
import argparse,hashlib,json,shutil

parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
root=Path(__file__).resolve().parent;out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
source=(root/'probes/r00_wgmma.cu').read_text()
source=source.replace('value != "batch16_wait1" && value != "single_wait0"','value != "cfg_c_batch8_wait1"')
source=source.replace('--mode must be batch16_wait1 or single_wait0','--mode must be cfg_c_batch8_wait1')
source=source.replace('serialized = value == "single_wait0";','serialized = false;')
source=source.replace('constexpr int Batch = Serialized ? 1 : 16;','constexpr int Batch = 4; // Two chains: eight MMA instructions per WG group.')
start=source.index('  for (int i = 0; i < 30; ++i) {');end=source.index('  launch();\n  uint64_t elapsed',start)
source=source[:start]+'''  for (int i = 0; i < 16; ++i) {
    launch();
    warmup.push_back(float(window()));
  }
  converged = coefficient_of_variation(std::vector<float>(warmup.end()-5,warmup.end())) <= 0.02;
'''+source[end:]
needle='  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));'
source=source.replace(needle,needle+'''
  char gpu_uuid[41] = "GPU-";
  int uuid_pos = 4;
  for (int i=0; i<16; ++i) {
    if (i==4 || i==6 || i==8 || i==10) gpu_uuid[uuid_pos++]='-';
    std::snprintf(gpu_uuid+uuid_pos,3,"%02x",unsigned(uint8_t(prop.uuid.bytes[i])));
    uuid_pos+=2;
  }
  if (std::string(gpu_uuid)!="GPU-ef8692f3-8bd7-80a2-608b-10731c9ec114")
    throw std::runtime_error("fixed reference GPU required");
''')
source=source.replace('<< ",\\\"work_flop\\\":" << work', '<< ",\\\"gpu_uuid\\\":\\\"" << gpu_uuid << "\\\",\\\"warmup_calls\\\":16,\\\"warmup_policy\\\":\\\"fixed16_same_probe\\\""\n            << ",\\\"work_flop\\\":" << work')
start=source.index('int main(int argc, char** argv)')
source=source[:start]+'''int main(int argc, char** argv) {
  try {
    WOptions o(argc,argv);
    if (o.dtype!="fp16" || o.source!="SS" || o.n!=128 || o.groups!=2 ||
        o.b_major!="MN" || o.scope!="one_cta" || (o.iterations!=64 && o.iterations!=1024))
      throw std::runtime_error("fixed cfg_c instruction reference contract required");
    return run_wgmma<__half, G::MMA_64x128x16_F32F16F16_SS<G::Major::K,G::Major::MN>,
                       128,16,2,false,true>(o);
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\\n';
    return 1;
  }
}
'''
(out/'cfg_c_mma_reference.cu').write_text(source)
shutil.copyfile(root/'probes/r00_common.hpp',out/'r00_common.hpp')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
contract=dict(protocol='cfg-c-mma-reference-v1',source_r00_sha256=sha(root/'probes/r00_wgmma.cu'),files={p.name:sha(p) for p in out.iterdir()},instructions_per_wg_group=8,chains_per_wg=2,warpgroups=2,k_per_group=64,iterations=[64,1024],wait_groups=1,final_wait_groups=0,warmup_calls=16,gpu_uuid='GPU-ef8692f3-8bd7-80a2-608b-10731c9ec114',limits=['Resident reused SMEM operands; no TMA, producer, stage ring or cluster traffic.','Window includes initial/final CTA synchronization and accumulator checksum/drain after wait0; not identical to GEMM FIRST/MAIN.','64 groups matches MMA count;1024 groups sustained reference, no intercept fit.','Short coordinate checks preserved; formal uniform input differs from GEMM dyadic seed17.','Only prepares source; compilation/numerics/timing remain unverified.'])
(out/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
