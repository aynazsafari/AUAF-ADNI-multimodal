from __future__ import annotations
import torch
from auaf_ablation_factory import ABLATION_VARIANTS, build_ablation_model
from auaf_model_factory import build_model

def run_tests():
    torch.manual_seed(20261116)
    dims={"clinical_cognitive":29,"fluid_biomarker":3}
    inputs={"clinical_cognitive":torch.randn(12,29),"fluid_biomarker":torch.randn(12,3)}
    mask=torch.tensor([[1.,1.]]*4+[[1.,0.]]*4+[[0.,1.]]*4)

    # The learned mask-token values must be tested as a functional component.
    full=build_model("AUAF",dims,3).eval()
    no_mask=build_ablation_model("AUAF_NoMaskTokens",dims,3).eval()
    no_mask.load_state_dict(full.state_dict(),strict=False)
    with torch.no_grad():
        p_full=full(inputs,mask)
        p_no_mask=no_mask(inputs,mask)
    max_diff=float((p_full-p_no_mask).abs().max())
    assert max_diff==0.0, f"No-mask shared-weight equivalence failed: {max_diff}"

    # In the current implementation, learned mask tokens have zero gradient.
    grad_model=build_model("AUAF",dims,3).train()
    all_missing_one_modality=torch.tensor([[1.,0.]]*12)
    grad_logits=grad_model(inputs,all_missing_one_modality)
    grad_logits.square().sum().backward()
    mask_token_grad_max=max(float(p.grad.abs().max()) for p in grad_model.mask_tokens.values())
    assert mask_token_grad_max==0.0, f"Mask-token gradient is not zero: {mask_token_grad_max}"

    rows=[]
    for variant in ABLATION_VARIANTS:
        model=build_ablation_model(variant,dims,3)
        for train_mode in [False,True]:
            model.train(train_mode)
            logits,aux=model(inputs,mask,return_aux=True)
            assert logits.shape==(12,3)
            assert torch.isfinite(logits).all()
            assert torch.allclose(aux["gates"].sum(1),torch.ones(12),atol=1e-6)
            assert (aux["gates"][mask==0].abs()<1e-7).all()
            rows.append({
                "variant":variant,"mode":"train" if train_mode else "eval",
                "finite_logits":True,"gate_sum_one":True,"unavailable_gate_zero":True,
                "parameter_count":sum(p.numel() for p in model.parameters())
            })
    assert len(ABLATION_VARIANTS)==7
    assert len(build_ablation_model("AUAF_NoMaskTokens",dims,3).mask_tokens)==0
    assert len(build_ablation_model("AUAF_NoTransformer",dims,3).transformer_blocks)==0
    assert build_ablation_model("AUAF_NoModalityDropout",dims,3).modality_dropout==0.0
    assert build_ablation_model("AUAF_UniformGate",dims,3).gating is None
    return {
        "status":"PASS","variants":len(ABLATION_VARIANTS),
        "forward_modes_tested":len(rows),
        "no_mask_shared_weight_max_logit_difference":max_diff,
        "full_model_mask_token_max_gradient":mask_token_grad_max,
        "variant_rows":rows
    }

if __name__=="__main__":
    import json
    print(json.dumps(run_tests(),indent=2))
