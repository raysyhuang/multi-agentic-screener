#!/usr/bin/env python3
"""Independent reranking and screening of Claude's 2026-11-12 US input audit.
Uses only the frozen shared pack plus explicit corrections documented in the review.
It does not import Claude's code or mutate any frozen inputs.
"""
import csv, gzip, collections, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
PACK = ROOT / 'shared' / 'inputs-pack-full-20261109'
OUT = pathlib.Path(__file__).resolve().parent

def read(name):
    p = PACK / name
    op = gzip.open if name.endswith('.gz') else open
    with op(p, 'rt', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))

def write(name, rows, fields=None):
    p=OUT/name
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]))
        w.writeheader(); w.writerows(rows)

sessions=sorted(r['session'] for r in read('trading_sessions.csv'))
month_days=collections.defaultdict(list)
for d in sessions: month_days[d[:7]].append(d)
months=sorted(max(v) for v in month_days.values())
mi={d:i for i,d in enumerate(months)}
elig=collections.defaultdict(set)
for r in read('eligibility_by_signal.csv.gz'):
    if r['eligible_final']=='True': elig[r['signal_date']].add(r['engine_ticker'])
px={(r['engine_ticker'],r['month_end']):float(r['rank_price']) for r in read('rank_prices_month_end.csv.gz') if r['rank_price']}
ov=collections.defaultdict(list)
for r in read('rank_price_overrides.csv'):
    ov[r['engine_ticker'],r['month_end']].append((r['signal_from'],r['signal_to'],float(r['rank_price'])))
actions=collections.defaultdict(list)
for r in read('corporate_actions_applied.csv'): actions[r['engine_ticker']].append(r)

# Explicit nine Claude-audit corrections plus Queen's newly identified vendor double-adjustments.
# ARNC/HWM remove Massive's 0.801 factor before the separate pack spin factor; EQT
# removes Massive's 0.54437 factor before the pack factor. EQT/HWM are latent in these dates.
SCALE={
 ('GOOGL','2014-03-31'):0.5005005005005005,
 ('GOOG','2014-03-31'):0.5005005005005005,
 ('TMK','2014-06-30'):2/3,
 ('DISCA','2014-07-31'):1000/1957,
 ('BF.B','2016-07-29'):0.5,
 ('BLL','2017-04-28'):0.5,
 ('BF.B','2018-02-28'):0.8,
 ('BKNG','2018-01-31'):0.04,
 ('ARNC','2016-10-31'):1/0.801,
 ('HWM','2016-10-31'):1/0.801,
 ('EQT','2018-10-31'):1/0.54437,
}

def calc(corrected=False):
    # Corrected IR/TT identity: old IR history belongs to today's TT; the merged
    # Gardner Denver issuer's new IR series starts Mar-2020 and is not seasoned.
    eligible={s:set(v) for s,v in elig.items()}
    prices=dict(px)
    if corrected:
        for (t,through),factor in SCALE.items():
            for (name,d),p in px.items():
                if name==t and d<=through: prices[name,d]=prices[name,d]*factor
        for (t,d),p in px.items():
            if t=='IR' and d<='2020-02-28': prices['TT',d]=p*(1/1.289)
        for s in eligible:
            if '2020-03-31'<=s<='2021-02-26':
                eligible[s].discard('IR'); eligible[s].add('TT')
    def price(t,d,s):
        for a,b,val in ov.get((t,d),[]):
            if a<=s<=b:return val
        return prices.get((t,d))
    result={}
    affected=[]
    for s,tickers in sorted(eligible.items()):
        if s not in mi: continue
        for w in range(1,13):
            b=months[mi[s]-w]
            picks=[]; invalid=[]
            for t in tickers:
                p0,p1=price(t,b,s),price(t,s,s)
                factor=1.0; bad=False
                if p0 is None or p1 is None or p0<=0 or p1<=0: bad=True
                for a in actions[t]:
                    if a['kind'] in ('spin','special_div') and b<a['ex_date']<=s:
                        if not a['factor_F']:bad=True
                        else:factor*=float(a['factor_F'])
                    elif a['kind']=='unresolved_BLOCKER' and b<a['ex_date']<=s:bad=True
                    elif a['kind']=='legacy_data_error_BLOCKER' and a['blocker_from']<=s<=a['blocker_to']:bad=True
                if bad:invalid.append(t)
                else:picks.append((t,p1/(p0*factor)-1))
            picks.sort(key=lambda z:(-z[1],z[0]))
            result[s,w]=(invalid,picks)
    return result

base=calc(False); fix=calc(True)
rows=[]; affected=[]; corrected_rank_rows=[]
for key in sorted(base):
    bad0,p0=base[key]; bad1,p1=fix[key]
    top0=[t for t,r in p0[:3]]; top1=[t for t,r in p1[:3]]
    top10_0=[t for t,r in p0[:10]]; top10_1=[t for t,r in p1[:10]]
    rows.append({'signal_date':key[0],'W':key[1],'base_status':'UNKNOWN' if bad0 else 'RANKED',
                 'corrected_status':'UNKNOWN' if bad1 else 'RANKED','base_top3':'|'.join(top0),
                 'corrected_top3':'|'.join(top1),'base_top10':'|'.join(top10_0),'corrected_top10':'|'.join(top10_1),'top3_changed':top0!=top1,
                 'top5_changed':[t for t,r in p0[:5]] != [t for t,r in p1[:5]],
                 'top10_changed':top10_0!=top10_1,
                 'base_unknown_names':'|'.join(sorted(bad0)), 'corrected_unknown_names':'|'.join(sorted(bad1))})
    d0={t:(rnk,r) for rnk,(t,r) in enumerate(p0,1)}; d1={t:(rnk,r) for rnk,(t,r) in enumerate(p1,1)}
    for t in sorted(set(d0)|set(d1)):
        x,y=d0.get(t),d1.get(t)
        if x==y or (x and y and abs(x[1]-y[1])<1e-12):
            continue
        affected.append({'signal_date':key[0],'W':key[1],'engine_ticker':t,
          'base_rank':x[0] if x else '', 'corrected_rank':y[0] if y else '',
          'base_return':format(x[1],'.15g') if x else '', 'corrected_return':format(y[1],'.15g') if y else ''})
write('rank_comparison_all_fixes_1584.csv',rows)
write('affected_rank_rows_all_fixes.csv',affected)
for (signal,w),(invalid,picks) in sorted(fix.items()):
    row={'signal_date':signal,'W':w,'eligible_count':len(invalid)+len(picks),'unrankable_count':len(invalid),'unrankable_names':'|'.join(sorted(invalid)),'status':'UNKNOWN' if invalid else 'RANKED'}
    for i in range(3):
        row[f'top{i+1}']=picks[i][0] if len(picks)>i and not invalid else ''
        row[f'ret{i+1}']=format(picks[i][1],'.15g') if len(picks)>i and not invalid else ''
    corrected_rank_rows.append(row)
write('corrected_rankings_all_fixes_1584.csv',corrected_rank_rows)

# Check Claude's per-path/stop path reruns bytewise and summarize outcome deltas.
result_dir=ROOT/'claude'/'input-audit-20261112'/'results'
checks=[]
for a,b in [('paths_5184_pack_inputs.csv','paths_5184_corrected_inputs.csv'),
            ('stoploss_324_pack_inputs.csv','stoploss_324_corrected_inputs.csv')]:
    x=(result_dir/a).read_bytes(); y=(result_dir/b).read_bytes()
    checks.append({'file_a':a,'file_b':b,'bytes_a':len(x),'bytes_b':len(y),'byte_identical':x==y})
write('claude_rerun_file_checks.csv',checks)

print('tables',len(rows),'Top3 changes',sum(r['top3_changed'] for r in rows),
      'Top5 changes',sum(r['top5_changed'] for r in rows),'Top10 changes',sum(r['top10_changed'] for r in rows),
      'affected cells',len(affected),'base unknown tables',sum(r['base_status']=='UNKNOWN' for r in rows),
      'corrected unknown tables',sum(r['corrected_status']=='UNKNOWN' for r in rows))
for x in checks:print(x)
