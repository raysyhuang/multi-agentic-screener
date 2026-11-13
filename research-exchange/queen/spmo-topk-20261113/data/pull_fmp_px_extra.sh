#!/bin/bash
# Pull FMP split-adjusted daily EOD (stable/historical-price-eod/full) for every symbol in data/symbols_extra.txt
# Requires env FMP_API_KEY. Output: data/fmp_px_2013_2026/<SYM>.json
cd "$(dirname "$0")/.."
one(){ s="$1"; f="data/fmp_px_2013_2026/${s}.json"; [ -s "$f" ] && exit 0
 curl -s --max-time 90 "https://financialmodelingprep.com/stable/historical-price-eod/full?symbol=${s}&from=2013-06-01&to=2026-11-12&apikey=$FMP_API_KEY" > "$f.tmp" && mv "$f.tmp" "$f"; }
export -f one
xargs -a data/symbols_extra.txt -P 8 -I{} bash -c 'one "$@"' _ {}
echo DONE
