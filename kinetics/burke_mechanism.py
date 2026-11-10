"""Convert the pinned public Burke release, preserving its H2/air kinetics.

The release contains 13 species and 27 rate entries. Remove the absent diluents
HE/CO/CO2 and explicit HE channels, rather than deleting active chemistry to
force agreement with the thesis's differently counted '20 reactions'.
"""
import json
from pathlib import Path
import hashlib

import cantera as ct
import numpy as np


KEEP = ['H2','H','O','O2','OH','H2O','HO2','H2O2','AR','N2']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(root=Path('third_party/burke_2012')):
    from cantera import cti2yaml
    root=Path(root)
    original=root/'h2_burke2012.cti'
    assert sha(original)=='ff4399a14e9bd2d19ba97a0d4add55b51a8503e0a3046d2444a9b53db0c2a888'
    full_path=root/'h2_burke2012_full.yaml'
    reduced_path=root/'h2_burke2012_H2_air10.yaml'
    if reduced_path.exists():
        raise FileExistsError('Do not overwrite a mechanism conversion')
    if not full_path.exists():
        cti2yaml.convert(str(original),output_name=full_path)
    full=ct.Solution(str(full_path))
    species=[full.species(s) for s in KEEP]
    thermo=ct.Solution(thermo='ideal-gas',kinetics='gas',species=species)
    retained=[];removed=[]
    for reaction in full.reactions():
        # Cantera removes explicit collider HE from the reactant/product dicts.
        # Check the third-body name too, before parsing against reduced species.
        collider=reaction.third_body.name if reaction.third_body is not None else 'M'
        if not (set(reaction.reactants)|set(reaction.products))<=set(KEEP) or (collider!='M' and collider not in KEEP):
            removed.append(reaction.equation);continue
        data=dict(reaction.input_data)
        if 'efficiencies' in data:
            data['efficiencies']={s:v for s,v in data['efficiencies'].items() if s in KEEP}
        retained.append(ct.Reaction.from_dict(data,thermo))
    reduced=ct.Solution(thermo='ideal-gas',kinetics='gas',species=species,reactions=retained)
    reduced.write_yaml(reduced_path,header=False)
    # Removing unpopulated inert species must not change production rates,
    # thermodynamics or molecular properties for the retained composition.
    rng=np.random.default_rng(111)
    checks=[]
    for i in range(20):
        T=float(rng.uniform(950,3000));P=float(rng.uniform(.5,5)*ct.one_atm)
        y=rng.uniform(.001,1,len(KEEP));y[KEEP.index('AR')]=0;y/=y.sum()
        composition=dict(zip(KEEP,y))
        full.TPY=T,P,composition;reduced.TPY=T,P,composition
        a=full.net_production_rates[[full.species_index(s) for s in KEEP]]
        b=reduced.net_production_rates
        np.testing.assert_allclose(a,b,rtol=1e-11,atol=1e-12)
        np.testing.assert_allclose(full.enthalpy_mass,reduced.enthalpy_mass,rtol=1e-13)
        checks.append(dict(T=T,P_Pa=P,maximum_rate_difference=float(np.max(np.abs(a-b)))))
    result=dict(status='passed',original_species=full.n_species,original_rate_entries=full.n_reactions,
        retained_species=KEEP,retained_rate_entries=reduced.n_reactions,unique_printed_equations=len(set(reduced.reaction_equations())),
        removed_channels=removed,absent_species=['HE','CO','CO2'],rate_equivalence_checks=checks,
        note='Thesis reports 10 species/20 reactions. The exact author mechanism file is unavailable; the public release retains 25 rate entries after inert removal. No active reaction is deleted merely to match that count.',
        converted_full_sha256=sha(full_path),converted_reduced_sha256=sha(reduced_path),cantera=ct.__version__)
    (root/'conversion_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='rate_equivalence_checks'}))
    return result


if __name__=='__main__':
    build()
