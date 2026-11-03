"""Joint total/data/element plateau checks; keep the10k trainer immutable."""
from .deeponet_long_training import new_monitor,check_plateau

COMPONENTS={'total':'full_train_loss','data':'data_loss','element':'element_loss'}


def joint_monitor(policy,run_sha256,origin_update,origin_seconds):
    return dict(enabled=True,policy=policy,long_run_sha256=run_sha256,
        origin_update=origin_update,origin_seconds=origin_seconds,checks=[],plateau=False,
        components={name:new_monitor(policy,origin_update,origin_seconds) for name in COMPONENTS})


def joint_check(history,update,monitor):
    policy=monitor['policy']
    if update<policy['minimum_updates'] or update%policy['check_every']:return False
    if monitor['checks'] and monitor['checks'][-1]['update']==update:return monitor['plateau']
    passed={}
    for name,field in COMPONENTS.items():
        values=[dict(update=p['update'],full_train_loss=p[field]) for p in history if field in p]
        passed[name]=check_plateau(values,update,monitor['components'][name])
    monitor['plateau']=all(passed.values())
    monitor['checks'].append(dict(update=update,component_plateaus=passed,
        latest_passes={name:part['checks'][-1]['passed'] for name,part in monitor['components'].items()},
        plateau=monitor['plateau']))
    return monitor['plateau']
