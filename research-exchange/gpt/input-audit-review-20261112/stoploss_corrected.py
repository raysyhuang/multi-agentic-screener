#!/usr/bin/env python3
"""Close-based fixed 30% stop sensitivity for US W10-12/H2-4 momentum paths.

Research only. Uses the frozen 2026-11-09 input pack and GPT independent ranks.
Rule: for each economic holding, maintain a weighted-average purchase-price basis.
On a non-rebalance session, if close <= 70% of basis, sell at that close, pay
one 25bp fee, and hold proceeds as cash until the next scheduled rebalance.
At a scheduled rebalance the portfolio is re-ranked and reconstructed normally;
a stopped ticker may be bought again only if it is selected then. Stops are
close-only (no intraday low data), not trailing, and no cash yield/tax is modeled.

Run from repo root:
  python research-exchange/gpt/independent_ranker.py
  python research-exchange/gpt/stoploss_sensitivity_w10_12_h2_4.py
"""
import csv
import datetime as dt
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import independent_execution as base

FEE = base.FEE


def simulate(L, w, h, k, phase, stop=0.30):
    start = L.first_of_month[base.month_of_index(phase-1)]
    end = L.first_of_month[base.month_of_index(phase+119)]
    rebalances = {}
    for m in range(phase-1, phase+119, h):
        day = L.first_of_month[base.month_of_index(m)]
        if day >= end:
            break
        nxt = L.first_of_month.get(base.month_of_index(m+h))
        rebalances[day] = end if nxt is None or nxt >= end else nxt

    holdings, basis = {}, {}
    cash, prev_session = 1.0, start
    value, peak, mdd = 1.0, float('-inf'), 0.0
    trades = executed = stop_count = fallback_count = 0
    stopped_names = []
    selections = []

    def invalid(reason, ticker='', day=''):
        return dict(known=False, reason=reason, ticker=ticker, date=day,
                    multiple=None, cagr=None, mdd=None, trades=trades,
                    rebalances=executed, stops=stop_count, fallback_uses=fallback_count, stop_names='|'.join(stopped_names))

    for di in range(L.day_at[start], L.day_at[end]+1):
        day = L.sessions[di]
        # Corporate actions precede mark-to-market, following frozen SPEC.
        for econ, (qty, series, _) in list(holdings.items()):
            touched = [ev for ev in L.events[series, econ]
                       if prev_session < ev['ex_date'] <= day]
            unresolved = next((ev for ev in touched if ev['type'] == 'unresolved'), None)
            if unresolved:
                return invalid('corporate_action_unresolved', unresolved['event_id'], day)
            for ev in touched:
                if ev['action'] in ('skip', 'ignore'):
                    continue
                if ev['action'] == 'fail':
                    return invalid('corporate_action_unresolved', ev['event_id'], day)
                amount = base.float_or_zero(ev['cash_per_share'])
                ratio = base.float_or_zero(ev['ratio'])
                if ev['type'] in ('spin', 'special_div'):
                    child_per_share = ratio * base.float_or_zero(ev['child_first_close'])
                    cash += qty*amount + qty*child_per_share*(1-FEE)
                    basis[econ] = max(0.0, basis.get(econ, 0.0) - amount - child_per_share)
                elif ev['type'] == 'cash':
                    cash += qty*amount
                    holdings.pop(econ, None)
                    basis.pop(econ, None)
                elif ev['type'] == 'stock':
                    target, target_series = ev['child_or_acquirer'], ev['acquirer_series']
                    new_price = L.price(target_series, target, day)
                    if not new_price:
                        return invalid('corporate_action_unresolved', f'{econ}->{target}', day)
                    existing_qty, existing_basis = 0.0, 0.0
                    if target in holdings and target != econ:
                        q0, s0, _ = holdings[target]
                        p0 = L.price(s0, target, day)
                        if p0 is None:
                            return invalid('daily_gap', target, day)
                        existing_qty = q0*p0/new_price
                        existing_basis = basis.get(target, p0)
                    old_basis = basis.pop(econ, 0.0)
                    transformed = max(0.0, (old_basis-amount)/ratio) if ratio > 0 else 0.0
                    new_qty = qty*ratio
                    cash += qty*amount
                    holdings.pop(econ, None)
                    basis.pop(target, None)
                    total_qty = existing_qty + new_qty
                    basis[target] = ((existing_qty*existing_basis + new_qty*transformed)/total_qty
                                     if total_qty else new_price)
                    holdings[target] = (total_qty, target_series, new_price)

        value = cash
        for econ, (qty, series, _) in list(holdings.items()):
            px = L.price(series, econ, day)
            if px is None:
                last = L.last.get((series, econ), '9999-12-31')
                return invalid('corporate_action_unresolved' if day > last else 'daily_gap', econ, day)
            holdings[econ] = (qty, series, px)
            value += qty*px

        if day in rebalances:
            signal = L.sessions[di-1]
            rank = L.ranking[signal, w]
            if rank['status'] == 'UNKNOWN':
                return invalid('rank_missing_unbounded', rank['unrankable_names'], day)
            names = [rank[f'top{i+1}'] for i in range(k)]
            keys = [base.execution_key(t, day) for t in names]
            if len(set(keys)) != len(keys):
                return invalid('duplicate_economic_execution_key', '|'.join(names), day)
            series_for = {}
            for ticker, ek in zip(names, keys):
                if ek in holdings:
                    series_for[ek] = holdings[ek][1]
                    continue
                trade = L.buy_data.get((day, ticker, rebalances[day]))
                if trade is None:
                    # Reuse only an unambiguous verified same-day series from
                    # another documented holding window; require full daily
                    # close coverage through this exact window end.
                    peers = [r for (d0,t0,_),r in L.buy_data.items()
                             if d0 == day and t0 == ticker]
                    series_set = {r['price_series'] for r in peers if r['status'] == 'ok'}
                    peer_statuses = {r['status'] for r in peers}
                    ek = base.execution_key(ticker, day)
                    peers_match_close = all(
                        L.price(r['price_series'], ek, day) is not None
                        and abs(float(r['exec_close'])-L.price(r['price_series'], ek, day)) <= 1e-8
                        for r in peers if r['status'] == 'ok')
                    candidates = [s for s in series_set
                                  if peer_statuses == {'ok'} and peers_match_close
                                  and L.last.get((s, base.execution_key(ticker, day)), '') >= rebalances[day]
                                  and L.price(s, base.execution_key(ticker, day), day) is not None
                                  and all(L.price(s, base.execution_key(ticker, day), L.sessions[j]) is not None
                                          for j in range(L.day_at[day], L.day_at[rebalances[day]]+1))]
                    if len(candidates) == 1:
                        trade = {'status':'ok', 'price_series':candidates[0]}
                        fallback_count += 1
                if trade is None:
                    return invalid('NOT_IN_PACK', ticker, day)
                if trade['status'] == 'not_tradable':
                    parts = trade['detail'].split('@')
                    return invalid('pick_not_tradable', parts[0], parts[-1] if len(parts)>1 else day)
                if trade['status'] != 'ok':
                    return invalid('exec_price_missing', ticker, day)
                series_for[ek] = trade['price_series']
            before = {ek: q*p for ek,(q,_,p) in holdings.items()}
            ep = value
            for _ in range(200):
                target_value = ep/k
                turnover = sum(abs((target_value if e in keys else 0.0)-before.get(e,0.0))
                                   for e in (set(keys)|set(before)))
                nxt = value-FEE*turnover
                if abs(nxt-ep) <= 1e-12*value:
                    ep = nxt
                    break
                ep = nxt
            target_value = ep/k
            updated, new_basis = {}, {}
            for ek in keys:
                series = series_for[ek]
                px = L.price(series, ek, day)
                if not px:
                    return invalid('exec_price_missing', ek, day)
                new_qty = target_value/px
                old_qty = holdings.get(ek, (0.0,'',0.0))[0]
                old_basis = basis.get(ek, px)
                if new_qty > old_qty:
                    blended = ((old_qty*old_basis)+(new_qty-old_qty)*px)/new_qty
                else:
                    blended = old_basis
                updated[ek] = (new_qty, series, px)
                new_basis[ek] = blended
                if abs(new_qty-old_qty) > 1e-12*max(old_qty,1e-30):
                    trades += 1
            trades += sum(ek not in updated for ek in holdings)
            holdings, basis, cash, value = updated, new_basis, 0.0, ep
            selections.extend((signal, day, ticker) for ticker in names)
            executed += 1
        else:
            # Fixed 30% close stop from weighted-average purchase-price basis.
            for econ, (qty, series, px) in list(holdings.items()):
                cost = basis.get(econ, px)
                if stop is not None and cost > 0 and px <= cost*(1-stop):
                    cash += qty*px*(1-FEE)
                    trades += 1
                    stop_count += 1
                    stopped_names.append(econ)
                    holdings.pop(econ, None)
                    basis.pop(econ, None)
            value = cash + sum(q*px for q,_,px in holdings.values())

        if day == end:
            stock_value = sum(q*px for q,_,px in holdings.values())
            value = cash + (1-FEE)*stock_value
            trades += len(holdings)
        peak = max(peak, value)
        mdd = min(mdd, value/peak-1.0)
        prev_session = day

    if k == 1:
        for signal, day, ticker in selections:
            if any(signal < d <= day for d in L.delist.get(ticker, [])):
                return invalid('pick_not_tradable', ticker, day)
    years = (dt.date.fromisoformat(end)-dt.date.fromisoformat(start)).days/365.25
    return dict(known=True, reason='', ticker='', date='', multiple=value,
                cagr=value**(1/years)-1, mdd=mdd, trades=trades,
                rebalances=executed, stops=stop_count, fallback_uses=fallback_count, stop_names='|'.join(stopped_names))


def main():
    L = base.Ledger()
    queen = {r['path_id']: r for r in base.read(base.QUEEN/'paths_summary.csv')}
    rows=[]
    for w in (10,11,12):
        for h in (2,3,4):
            for k in (1,2,3):
                for phase in range(1,13):
                    ident=f'W{w}H{h}K{k}P{phase}'
                    r=simulate(L,w,h,k,phase,stop=0.30)
                    control=simulate(L,w,h,k,phase,stop=None)
                    q=queen[ident]
                    r['path_id']=ident; r['W']=w; r['H']=h; r['K']=k; r['phase']=phase
                    r['base_status']=q['status']; r['base_CAGR']=q['CAGR']; r['base_MDD']=q['daily_MDD']
                    r['base_final_multiple']=q['final_multiple']
                    r['control_CAGR']=control['cagr']; r['control_MDD']=control['mdd']; r['control_multiple']=control['multiple']
                    rows.append(r)
    out=HERE/'stoploss_w10_12_h2_4_30pct_paths.csv'
    cols=['path_id','W','H','K','phase','known','reason','ticker','date','multiple','cagr','mdd','trades','rebalances','stops','fallback_uses','stop_names','base_status','base_CAGR','base_MDD','base_final_multiple','control_CAGR','control_MDD','control_multiple']
    with out.open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=cols);wr.writeheader();wr.writerows(rows)
    print('paths',len(rows),'stop known',sum(r['known'] for r in rows),'UNKNOWN',sum(not r['known'] for r in rows))
    print('control max abs diffs vs Queen', max(abs(float(r['control_CAGR'])-float(r['base_CAGR'])) for r in rows), max(abs(float(r['control_MDD'])-float(r['base_MDD'])) for r in rows), max(abs(float(r['control_multiple'])-float(r['base_final_multiple'])) for r in rows))
    print('fallback re-entry series uses',sum(int(r['fallback_uses']) for r in rows))
    # Per-cell 12-phase summary (median uses conventional average of the two middle values).
    summary=[]
    for w in (10,11,12):
        for h in (2,3,4):
            for k in (1,2,3):
                xs=[r for r in rows if r['W']==w and r['H']==h and r['K']==k]
                for label,key in [('stop','cagr'),('base','base_CAGR')]:
                    vals=[float(x[key]) for x in xs]
                    md=[float(x['mdd'] if label=='stop' else x['base_MDD']) for x in xs]
                    summary.append(dict(W=w,H=h,K=k,mode=label,n=len(xs),median_CAGR=statistics.median(vals),min_CAGR=min(vals),max_CAGR=max(vals),median_MDD=statistics.median(md),worst_MDD=min(md),stop_events=sum(int(x['stops']) for x in xs),paths_with_stop=sum(int(x['stops'])>0 for x in xs)))
    sout=HERE/'stoploss_w10_12_h2_4_30pct_summary.csv'
    with sout.open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=list(summary[0]));wr.writeheader();wr.writerows(summary)
    print('wrote',out.name,'and',sout.name)
    for k in (1,2,3):
        xs=[r for r in rows if r['K']==k and r['known']]
        print('K',k,'n',len(xs),'median CAGR',sorted(r['cagr'] for r in xs)[len(xs)//2],
              'median MDD',sorted(r['mdd'] for r in xs)[len(xs)//2],
              'median stops',sorted(r['stops'] for r in xs)[len(xs)//2])

if __name__=='__main__': main()
