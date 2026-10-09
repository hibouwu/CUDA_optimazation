"""Cooperative output intervals from valid-output fractions and calibrated rules."""


def output_intervals(params, schedule, valid, first_bytes, single, output_rule):
    """Return copied CTA parameters and optional per-tile merged epilogues in ns.

    first_bytes is the total valid first-output byte count across all CTAs.
    single describes the maximum scheduled tile count across the whole grid.
    """
    p = dict(params)
    if not output_rule or not valid:
        return p, None
    if schedule != 'cooperative':
        raise ValueError('these output rules were calibrated for cooperative kernels')
    if single:
        rule = output_rule['E0_single']
        e0 = max(rule['floor_ns'], rule['beta_ns_per_MiB']*first_bytes/2**20) + rule['R_ns']
        p['Etail'] = output_rule['Etail_single_mean_ns']
    else:
        e0 = output_rule['E0_multi']['effective_ns']
        p['Etail'] = ((1-valid[-1])*output_rule['Etail_oob_median_ns']
                      + valid[-1]*output_rule['v09_Etail_multi_median_ns'])
    middle = output_rule['E_middle_ns']
    if middle is None:
        middle = output_rule['middle_padding_proxy']['ns']
    epilogues = [e0]
    for j, fraction in enumerate(valid[1:], 1):
        key = 'Elast' if j == len(valid)-1 else 'E_middle'
        full = output_rule['Elast_ns'] if key == 'Elast' else middle
        epilogues.append((1-fraction)*output_rule[key+'_oob_ns'] + fraction*full)
    return p, epilogues
