import json
import pandas as pd
import numpy as np
import re
import unicodedata
import difflib
import os
import time
from itertools import combinations
from scipy.optimize import minimize
from scipy.stats import poisson
from curl_cffi import requests  # Bypasses the firewall

# ============================================================
# CONFIG
# ============================================================
BASE_URL = "https://www.sportybet.com"
REGION = "ke"
CACHE_FILE = "football_cache.json"

LEAGUES = {
    "EPL": "en.1", "Championship": "en.2", "La Liga": "es.1", "Segunda": "es.2",
    "Bundesliga": "de.1", "2. Bundesliga": "de.2", "Serie A": "it.1", "Serie B": "it.2",
    "Ligue 1": "fr.1", "Ligue 2": "fr.2", "Eredivisie": "nl.1", "Primeira Liga": "pt.1",
    "Belgian Pro League": "be.1", "Süper Lig": "tr.1", "Russian Premier League": "ru.1",
    "Austrian Bundesliga": "at.1", "Swiss Super League": "ch.1", "Super League Greece": "gr.1",
    "Scottish Premiership": "scotland", "Ukrainian Premier League": "ua.1", "Liga I": "ro.1",
    "Ekstraklasa": "pl.1", "Danish Superliga": "dk.1", "Allsvenskan": "se.1", "Eliteserien": "no.1",
    "Czech First League": "cz.1", "HNL": "hr.1", "SuperLiga": "rs.1", "Serie A (Brazil)": "br.1",
    "Primera División (Argentina)": "ar.1", "Liga MX": "mx.1", "MLS": "us.1", "J1 League": "jp.1",
}

# ============================================================
# TEAM RESOLVER & AI MODEL
# ============================================================
NOISE_WORDS = {"fc","afc","cf","sc","ac","as","ss","ssc","sv","vfl","vfb","bsc","tsg",
    "fsv","bk","sk","fk","kf","nk","hnk","cd","sd","ud","rc","rcd","cs","us","club",
    "de","futbol","football","sport","sportif","calcio","soccer","the",
    "1899","1900","1901","1904","1907","1909","1911","1913","04","05","07","09","96","98"}

TEAM_ALIASES = {
    "man city":"manchester city","man utd":"manchester united","man united":"manchester united",
    "newcastle":"newcastle united","west ham":"west ham united","brighton":"brighton & hove albion",
    "tottenham":"tottenham hotspur","spurs":"tottenham hotspur","wolves":"wolverhampton wanderers",
    "nottm forest":"nottingham forest","nott'm forest":"nottingham forest","nottingham":"nottingham forest",
    "sheffield utd":"sheffield united","sheffield weds":"sheffield wednesday","leicester":"leicester city",
    "leeds":"leeds united","norwich":"norwich city","stoke":"stoke city","swansea":"swansea city",
    "hull":"hull city","cardiff":"cardiff city","birmingham":"birmingham city","barca":"barcelona",
    "atletico":"atletico madrid","real":"real madrid","betis":"real betis","sociedad":"real sociedad",
    "athletic club":"athletic bilbao","celta":"celta vigo","rayo":"rayo vallecano","bayern":"bayern munich",
    "dortmund":"borussia dortmund","bvb":"borussia dortmund","leverkusen":"bayer leverkusen",
    "leipzig":"rb leipzig","frankfurt":"eintracht frankfurt","wolfsburg":"vfl wolfsburg",
    "gladbach":"borussia monchengladbach","freiburg":"sc freiburg","hoffenheim":"tsg hoffenheim",
    "mainz":"mainz 05","augsburg":"fc augsburg","stuttgart":"vfb stuttgart","werder":"werder bremen",
    "union berlin":"union berlin","bochum":"vfl bochum","heidenheim":"1. fc heidenheim",
    "darmstadt":"sv darmstadt 98","koln":"1. fc koln","inter":"inter milan","milan":"ac milan",
    "juve":"juventus","roma":"as roma","lazio":"ss lazio","verona":"hellas verona",
    "psg":"paris saint-germain","marseille":"olympique marseille","lyon":"olympique lyonnais",
    "monaco":"as monaco","lens":"rc lens","nice":"ogc nice","rennes":"stade rennais",
    "reims":"stade de reims","strasbourg":"rc strasbourg","brest":"stade brestois",
    "metz":"fc metz","lorient":"fc lorient","clermont":"clermont foot","az":"az alkmaar",
    "twente":"fc twente","utrecht":"fc utrecht","sporting":"sporting cp","braga":"sc braga",
    "fenerbahce":"fenerbahce","besiktas":"besiktas","sao paulo":"sao paulo","gremio":"gremio",
    "vasco":"vasco da gama","boca":"boca juniors","river":"river plate","lafc":"los angeles fc",
    "america":"club america","chivas":"guadalajara","tigres":"tigres uanl",
}

def normalize(name):
    if not name: return ""
    s = name.lower().strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii")
    s = re.sub(r"[^\w\s]"," ",s)
    s = re.sub(r"\s+"," ",s).strip()
    return " ".join(t for t in s.split() if t not in NOISE_WORDS).strip()

class TeamResolver:
    def __init__(self, known_teams):
        self.known = list(known_teams)
        self.known_norm = {}
        for t in self.known: self.known_norm.setdefault(normalize(t), t)
        self.alias_norm = {normalize(k): normalize(v) for k, v in TEAM_ALIASES.items()}
        
    def resolve(self, name, cutoff=0.70):
        if not name: return None
        if name.strip() in self.known: return name.strip()
        norm = normalize(name)
        if not norm: return None
        if norm in self.known_norm: return self.known_norm[norm]
        alias = self.alias_norm.get(norm)
        if alias:
            if alias in self.known_norm: return self.known_norm[alias]
            at = set(alias.split())
            for kn, kr in self.known_norm.items():
                kt = set(kn.split())
                if at and (at <= kt or kt <= at): return kr
        nt = set(norm.split()); best, best_s = None, 0.0
        for kn, kr in self.known_norm.items():
            kt = set(kn.split())
            if kt and (nt <= kt or kt <= nt):
                s = len(nt & kt)/max(len(nt | kt),1)
                if s > best_s: best_s, best = s, kr
        if best and best_s >= 0.4: return best 
        for kn, kr in self.known_norm.items():
            if len(kn) > 3 and (kn in norm or norm in kn): return kr
        m = difflib.get_close_matches(norm, list(self.known_norm.keys()), n=1, cutoff=cutoff)
        return self.known_norm[m[0]] if m else None

class DixonColes:
    def __init__(self):
        self.teams, self.idx = [], {}
        self.attack, self.defense = {}, {}
        self.home_adv, self.rho = 0.0, 0.0
        self.resolver = None

    def _tau(self, x, y, lam, mu, rho):
        if x==0 and y==0: return 1 - lam*mu*rho
        if x==0 and y==1: return 1 + lam*rho
        if x==1 and y==0: return 1 + mu*rho
        if x==1 and y==1: return 1 - rho
        return 1.0

    def _nll(self, params, h_idx, a_idx, hg, ag, w):
        n = len(self.teams)
        att = params[:n] - params[:n].mean()
        dfn = params[n:2*n]
        ha = params[2*n]; rho = params[2*n+1]
        lam = np.exp(att[h_idx] + dfn[a_idx] + ha)
        mu  = np.exp(att[a_idx] + dfn[h_idx])
        tau = np.ones_like(lam)
        tau = np.where((hg == 0) & (ag == 0), 1 - lam * mu * rho, tau)
        tau = np.where((hg == 0) & (ag == 1), 1 + lam * rho, tau)
        tau = np.where((hg == 1) & (ag == 0), 1 + mu * rho, tau)
        tau = np.where((hg == 1) & (ag == 1), 1 - rho, tau)
        log_lik = np.log(np.maximum(tau, 1e-9)) + poisson.logpmf(hg, lam) + poisson.logpmf(ag, mu)
        ll = np.sum(w * log_lik)
        return -ll + 0.01*(np.sum(att**2)+np.sum(dfn**2))

    def fit(self, df):
        self.teams = sorted(set(df["home"])|set(df["away"]))
        self.idx = {t:i for i,t in enumerate(self.teams)}
        n = len(self.teams)
        latest = df["date"].max()
        w = np.exp(-0.0018*(latest-df["date"]).dt.days.values)
        h_idx = df["home"].map(self.idx).values
        a_idx = df["away"].map(self.idx).values
        hg = df["home_goals"].values
        ag = df["away_goals"].values
        x0 = np.concatenate([np.zeros(n),np.zeros(n),[0.25],[-0.05]])
        res = minimize(self._nll, x0, args=(h_idx, a_idx, hg, ag, w), 
                       method="L-BFGS-B", options={"maxiter":300})
        p = res.x; att = p[:n]-p[:n].mean(); dfn = p[n:2*n]
        self.attack = {t:att[self.idx[t]] for t in self.teams}
        self.defense = {t:dfn[self.idx[t]] for t in self.teams}
        self.home_adv, self.rho = p[2*n], p[2*n+1]
        self.resolver = TeamResolver(self.teams)
        return self

    def _resolve(self, n):
        if n in self.idx: return n
        return self.resolver.resolve(n) if self.resolver else None

    def predict(self, home, away, max_goals=8):
        hr = self._resolve(home); ar = self._resolve(away)
        if hr is None or ar is None: return None
        home, away = hr, ar
        lam = np.exp(self.attack[home]+self.defense[away]+self.home_adv)
        mu = np.exp(self.attack[away]+self.defense[home])
        m = np.zeros((max_goals+1,max_goals+1))
        for i in range(max_goals+1):
            for j in range(max_goals+1):
                m[i,j] = self._tau(i,j,lam,mu,self.rho)*poisson.pmf(i,lam)*poisson.pmf(j,mu)
        m /= m.sum(); tot = np.add.outer(np.arange(m.shape[0]), np.arange(m.shape[1]))
        return {"home":np.tril(m,-1).sum(),"draw":np.trace(m),"away":np.triu(m,1).sum(),
                "over15":m[tot>=2].sum(),"under15":m[tot<=1].sum(),
                "over25":m[tot>=3].sum(),"under25":m[tot<=2].sum(),
                "btts_yes":m[1:,1:].sum(),"btts_no":1-m[1:,1:].sum(),
                "1X":np.tril(m,-1).sum()+np.trace(m), "X2":np.trace(m)+np.triu(m,1).sum()}

# ============================================================
# DATA FETCHING & API
# ============================================================
def fetch_data():
    if not os.path.exists(CACHE_FILE):
        print("Downloading AI historical data (1-2 mins)...")
        rows = []
        for s in ["2024-25", "2023-24", "2022-23"]:
            for n, c in LEAGUES.items():
                url = f"https://raw.githubusercontent.com/openfootball/football.json/master/{s}/{c}.json"
                try:
                    r = requests.get(url, timeout=10, verify=False)
                    if r.status_code == 200:
                        for m in r.json().get("matches", []):
                            if m.get("score", {}).get("ft"):
                                rows.append({"date":m.get("date"),"home":m.get("team1"),"away":m.get("team2"),
                                             "home_goals":m["score"]["ft"][0],"away_goals":m["score"]["ft"][1]})
                except: pass
        df = pd.DataFrame(rows).dropna()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        cdf = df.copy(); cdf["date"] = cdf["date"].dt.strftime("%Y-%m-%d")
        with open(CACHE_FILE,"w") as f: json.dump(cdf.to_dict(orient="records"), f)
        return df
        
    with open(CACHE_FILE) as f: df = pd.DataFrame(json.load(f))
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"])

class SportyBetClient:
    def __init__(self): 
        self.session = requests.Session()
        self.session.verify = False
        
    def get_upcoming_games(self):
        print("Hunting for games on SportyBet...")
        try: self.session.get("https://www.sportybet.com/ke/", impersonate="chrome120", timeout=10)
        except: pass
        
        url = f"{BASE_URL}/api/{REGION}/factsCenter/pcUpcomingEvents"
        params = {"sportId": "sr:sport:1", "pageSize": 500, "pageNum": 1, "timeline": 48, "_t": int(time.time()*1000)}
        headers = {"Accept": "application/json, text/plain, */*", "Current-Country": "KE"}
        
        try:
            r = self.session.get(url, params=params, headers=headers, impersonate="chrome120", timeout=15)
            data = r.json()
            tournaments = data.get("data", {}).get("tournaments", [])
            
            events = []
            for t in tournaments:
                events.extend(t.get("events", []))
            return events
        except Exception as e:
            print(f"Error fetching: {e}")
            return []

    def create_booking_code(self, selections):
        def build_payload(sel_list):
            payload = {"selections": []}
            for s in sel_list:
                item = {"eventId": str(s["eventId"]), "marketId": str(s["marketId"]), "outcomeId": str(s["outcomeId"]), "productId": 3, "sportId": "sr:sport:1"}
                if s.get("specifier") and str(s["specifier"]).strip().lower() not in ["none", "null", ""]:
                    item["specifier"] = str(s["specifier"]).strip()
                payload["selections"].append(item)
            return payload

        if len(selections) > 20: selections = selections[:20]

        def try_generate(sel_list):
            if not sel_list: return None
            try:
                time.sleep(0.5) 
                r = self.session.post(f"{BASE_URL}/api/{REGION}/orders/share", json=build_payload(sel_list), impersonate="chrome120", timeout=15)
                if r.text.strip().startswith("<"): return None
                data = r.json()
                if data.get("bizCode") == 10000: return data.get("data",{}).get("shareCode")
            except: pass
            return None

        code = try_generate(selections)
        if code: return code

        print(f"      [RECOVERING] API rejected batch. Finding invalid games...")
        
        def find_valid_games(games):
            if not games: return []
            if len(games) == 1:
                code = try_generate(games)
                return games if code else []
            mid = len(games) // 2
            return find_valid_games(games[:mid]) + find_valid_games(games[mid:])

        valid_games = find_valid_games(selections)
        if len(valid_games) < len(selections):
            print(f"      [RECOVERING] Dropped {len(selections) - len(valid_games)} invalid games to bypass API block.")
            
        if valid_games: return try_generate(valid_games)
        return None

# ============================================================
# ALGORITHM LOGIC
# ============================================================
def analyze_upcoming(events, model):
    evaluated = []
    for ev in events:
        home = ev.get("homeTeamName") or "Unknown"
        away = ev.get("awayTeamName") or "Unknown"
        
        if "srl" in home.lower() or "srl" in away.lower(): continue

        p = model.predict(home, away)
        
        for m in ev.get("markets", []):
            m_desc = m.get("desc", "")
            for o in m.get("outcomes", []):
                odds = float(o.get("odds", 0))
                if odds <= 0: continue
                o_desc = o.get("desc", "")
                
                mp = None
                
                if p:
                    if "1X2" in m_desc:
                        if "Home" in o_desc or o_desc == "1": mp = p["home"]
                        elif "Draw" in o_desc or o_desc == "X": mp = p["draw"]
                        elif "Away" in o_desc or o_desc == "2": mp = p["away"]
                    elif "Over/Under" in m_desc:
                        if "Over 1.5" in o_desc: mp = p["over15"]
                        elif "Under 1.5" in o_desc: mp = p["under15"]
                        elif "Over 2.5" in o_desc: mp = p["over25"]
                        elif "Under 2.5" in o_desc: mp = p["under25"]
                    elif "GG/NG" in m_desc or "BTTS" in m_desc:
                        if "Yes" in o_desc or "GG" in o_desc: mp = p["btts_yes"]
                        elif "No" in o_desc or "NG" in o_desc: mp = p["btts_no"]
                    elif "Double Chance" in m_desc:
                        if "1X" in o_desc: mp = p["1X"]
                        elif "X2" in o_desc: mp = p["X2"]

                if mp is None or mp == 0.0:
                    mp = 1.0 / odds if odds > 1.0 else 0.5

                evaluated.append({
                    "match": f"{home} vs {away}", "home": home, "away": away,
                    "market": m_desc, "outcome": o_desc, "odds": odds, "prob": mp,
                    "eventId": ev.get("eventId"), "marketId": m.get("id"), "outcomeId": o.get("id"), "specifier": o.get("specifier", "")
                })
    return evaluated

def find_best_2_odds(picks):
    if not picks: return []
    
    # Sort by highest probability
    sorted_picks = sorted(picks, key=lambda x: x["prob"], reverse=True)
    best_combo = None
    best_score = -1
    
    print("Calculating best 2.0+ odds combinations...")
    
    # Try 1, 2, 3 or 4 games
    for size in range(1, 5):
        for combo in combinations(sorted_picks[:40], size):
            if len(set(x['eventId'] for x in combo)) < len(combo): continue
            
            odds = 1.0
            prob = 1.0
            for x in combo: 
                odds *= x["odds"]
                prob *= x["prob"]
            
            # 🚨 MUST BE 2.00 OR GREATER 🚨
            if odds >= 2.00 and odds <= 2.30:
                if prob > best_score:
                    best_score = prob
                    best_combo = combo
                    
    # Fallback: If no combination is exactly in the 2.00 - 2.30 range, find the closest >= 2.00
    if not best_combo:
        closest_dist = float('inf')
        for size in range(1, 6):
            for combo in combinations(sorted_picks[:20], size):
                if len(set(x['eventId'] for x in combo)) < len(combo): continue
                odds = 1.0
                prob = 1.0
                for x in combo: odds *= x["odds"]; prob *= x["prob"]
                
                if odds >= 2.00:
                    dist = odds - 2.00
                    if dist < closest_dist:
                        closest_dist = dist
                        best_combo = combo

    return best_combo or []

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print(" 🏹 SPORTYBET 2.0+ ODDS HUNTER (STRICT) 🏹")
    print("=" * 70)

    hist = fetch_data()
    model = DixonColes().fit(hist)
    client = SportyBetClient()

    events = client.get_upcoming_games()
    if not events:
        print("No upcoming games found or API blocked your IP.")
        exit()
        
    print(f"Found {len(events)} tournaments. Analyzing...\n")
    all_picks = analyze_upcoming(events, model)

    if not all_picks:
        print("No valid markets found.")
        exit()

    group = find_best_2_odds(all_picks)
    
    if group:
        o = 1.0; p_ = 1.0
        for s in group: o *= s["odds"]; p_ *= s["prob"]
        
        print("=" * 70)
        print(f" 🎯 BEST 2.0+ ODDS SLIP FOUND (Odds: {o:.2f} | Win Prob: {p_:.2%})")
        print("=" * 70)
        
        for s in group:
            print(f" • {s['match']}")
            print(f"   {s['outcome']} @ {s['odds']:.2f} (Prob: {s['prob']:.1%})")
        print("-" * 70)

        print("Generating SportyBet Code...")
        c = client.create_booking_code(group)
        if c: print(f" ✅ SPORTYBET CODE: {c}")
        else: print(" ❌ Failed to generate code (games might be suspended).")
    else:
        print("Could not find a high-probability combination at 2.00 odds or higher today.")

    print("\n" + "=" * 70)
    print(" 🏁 COMPLETE")
    print("=" * 70)