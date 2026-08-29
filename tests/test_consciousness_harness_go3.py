from nsa.cognition.substrate import CognitiveSwitches
from research.consciousness_harness import ConsciousnessResearchHarness, ExperimentConfig, paired_effect, reproducibility_record

def test_theory_cell_is_deterministic():
    h=ConsciousnessResearchHarness(); c=ExperimentConfig(theory="gwt",steps=5); obs=[1.,1.,2.,2.,1.]
    assert h.run_cell(c,obs)==h.run_cell(c,obs)

def test_ablation_matrix_has_full_and_single_controls():
    m=ConsciousnessResearchHarness.ablation_matrix(ExperimentConfig())
    assert len(m)==6 and m[0].switches==CognitiveSwitches() and not m[1].switches.workspace and not m[2].switches.recurrence and not m[3].switches.predictive_processing and not m[4].switches.self_model and not m[5].switches.integration

def test_paired_effect():
    r=paired_effect([1,2,3,4],[2,3,4,5]); assert r["mean_difference"]==1.0 and r["n"]==4.0

def test_reproducibility_record():
    r=ConsciousnessResearchHarness().run_cell(ExperimentConfig(theory="higher_order",steps=2),[0.,1.]); x=reproducibility_record(r)
    assert x["theory"]=="higher_order" and "self_model" in x["mechanisms"] and len(x["trajectory"])==2
