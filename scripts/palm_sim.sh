source /workspace/mp/scripts/env.sh
cd /workspace/mp/palm_det
sed 's#path: [^ ]*#path: /workspace/mp/palm_sim/dataset.txt#' net_inputmeta.yml > net_inputmeta_test.yml
for t in i16 pcq; do $PEG inference --model net.json --model-data net.data --dtype quantized --model-quantize net_$t.quantize --with-input-meta net_inputmeta_test.yml --device CPU --output-dir inf_$t --iterations 1 > inf_$t.log 2>&1; done
python3 - <<'P'
import glob,numpy as np
ref=np.load('/workspace/mp/palm_test/out/t_00.ref1.npy').ravel(); rr=np.load('/workspace/mp/palm_test/out/t_00.ref0.npy').ravel()
for t in ('i16','pcq'):
    o={}
    for f in glob.glob(f'/workspace/mp/palm_det/inf_{t}/iter_0_*.tensor'):
        if f.endswith('.qnt.tensor') or 'input' in f: continue
        a=np.loadtxt(f,dtype=np.float32).ravel(); o[a.size]=a
    print(t,{k:v.shape for k,v in o.items()})
    print(t,'score corr %.3f reg corr %.3f'%(np.corrcoef(o[2016],ref)[0,1],np.corrcoef(o[2016*18],rr)[0,1]))
P
chmod -R a+rwX /workspace/mp/palm_det
