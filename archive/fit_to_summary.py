import argparse
from fitparse import FitFile
import pandas as pd, numpy as np
from dateutil import tz
from datetime import timedelta
from pathlib import Path

# --- apu: NP ja TRIMP ---
def np_power(power):
    if len(power)==0: return np.nan
    s = pd.Series(power, dtype=float)
    # 1 s aikaleima oletus; 30 s liukuva KA
    ma30 = s.rolling(30, min_periods=1).mean()
    return float((ma30.pow(4).mean())**0.25)

def trimp_from_hr(hr, hr_rest=50, hr_max=190):
    if len(hr)==0: return 0.0
    s = pd.Series(hr, dtype=float).dropna()
    if s.empty: return 0.0
    hrr = (s - hr_rest)/max(1,(hr_max-hr_rest))
    hrr = hrr.clip(lower=0)
    inst = hrr * 0.64 * np.exp(1.92*hrr)
    minutes = len(inst)/60.0  # jos 1 s väli
    return float(inst.mean()*minutes)

def summarize_fit_sessions(path, ftp=300, hr_rest=50, hr_max=190, tz_name="Europe/Helsinki"):
    """
    Process each session in a FIT file separately to handle multisport activities correctly
    """
    ff = FitFile(path)

    # Get sessions first
    sessions = []
    for m in ff.get_messages("session"):
        d = {d.name: d.value for d in m}
        sessions.append(d)

    results = []

    # If no sessions or only one session, fall back to original behavior
    if len(sessions) <= 1:
        result, df_sets = summarize_fit_original(path, ftp, hr_rest, hr_max, tz_name)
        if result:
            results.append(result)
        return results, []

    # Process each session separately
    for session_idx, session in enumerate(sessions):
        session_start = session.get('start_time')
        session_timer_time = session.get('total_timer_time', 0)  # in seconds

        if session_start and session_timer_time > 0:
            session_end = session_start + timedelta(seconds=session_timer_time)

            # Get records for this session time window
            recs = []
            for m in ff.get_messages("record"):
                d = {d.name: d.value for d in m}
                if "timestamp" in d:
                    record_time = d["timestamp"]
                    # Check if record falls within this session's time window
                    if session_start <= record_time <= session_end:
                        recs.append({
                            "time": record_time,
                            "hr": d.get("heart_rate", np.nan),
                            "power": d.get("power", np.nan),
                        })

            # Process this session's data
            if recs:
                df = pd.DataFrame(recs).sort_values("time")
                session_summary = process_session_data(
                    df, path, session, session_idx, ftp, hr_rest, hr_max, tz_name
                )
                if session_summary:
                    results.append(session_summary)

    return results, []

def process_session_data(df, path, session, session_idx, ftp, hr_rest, hr_max, tz_name):
    """Process data for a single session"""
    if df.empty:
        return None

    local = tz.gettz(tz_name)
    start_utc = pd.to_datetime(df["time"].iloc[0]).tz_localize("UTC")
    end_utc = pd.to_datetime(df["time"].iloc[-1]).tz_localize("UTC")
    start_local = start_utc.astimezone(local)
    end_local = end_utc.astimezone(local)
    dur_sec = int((end_utc - start_utc).total_seconds()) + 1
    dur_hr = dur_sec/3600.0

    # Resample to 1 second for NP calculation
    df = df.set_index(pd.to_datetime(df["time"]).dt.tz_localize("UTC")).sort_index()
    df = df.resample("1s").ffill()

    avg_hr = float(df["hr"].mean()) if df["hr"].notna().any() else np.nan
    max_hr = float(df["hr"].max()) if df["hr"].notna().any() else np.nan
    avg_p = float(df["power"].mean()) if df["power"].notna().any() else np.nan
    max_p = float(df["power"].max()) if df["power"].notna().any() else np.nan
    npw = np_power(df["power"].fillna(0)) if df["power"].notna().any() else np.nan
    IF = (npw/ftp) if np.isfinite(npw) and ftp>0 else np.nan
    TSS = ((dur_hr * npw * IF)/ftp * 100) if np.all(np.isfinite([dur_hr,npw,IF])) and ftp>0 else np.nan
    TRIMP = trimp_from_hr(df["hr"].ffill(), hr_rest=hr_rest, hr_max=hr_max) if df["hr"].notna().any() else 0.0

    # Create filename with session info
    base_name = Path(path).stem
    session_sport = session.get('sport', 'unknown')
    session_subsport = session.get('sub_sport', '')

    if session_subsport:
        file_display = f"{base_name}_session{session_idx}_{session_sport}_{session_subsport}"
    else:
        file_display = f"{base_name}_session{session_idx}_{session_sport}"

    return {
        "file": file_display,
        "sport": session_sport,
        "sub_sport": session_subsport,
        "date": start_local.date().isoformat(),
        "start_time": start_local.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_local.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_min": round(dur_sec/60.0, 1),
        "avg_hr": round(avg_hr,1) if np.isfinite(avg_hr) else "",
        "max_hr": int(max_hr) if np.isfinite(max_hr) else "",
        "avg_power_w": round(avg_p,1) if np.isfinite(avg_p) else "",
        "max_power_w": round(max_p,1) if np.isfinite(max_p) else "",
        "np_w": round(npw,1) if np.isfinite(npw) else "",
        "IF": round(IF,3) if np.isfinite(IF) else "",
        "TSS": round(TSS,1) if np.isfinite(TSS) else "",
        "TRIMP": round(TRIMP,1),
        # Keep these for deduplication logic
        "_original_file": path,
        "_session_index": session_idx,
    }

def summarize_fit_original(path, ftp=300, hr_rest=50, hr_max=190, tz_name="Europe/Helsinki"):
    """Original function for single-session activities"""
    ff = FitFile(path)

    # Try to get session info for sport/sub_sport
    sessions = []
    for m in ff.get_messages("session"):
        d = {d.name: d.value for d in m}
        sessions.append(d)

    # Extract sport info from first session if available
    session_sport = sessions[0].get('sport', '') if sessions else ''
    session_subsport = sessions[0].get('sub_sport', '') if sessions else ''

    # --- record-viestit (aerobinen data) ---
    recs = []
    for m in ff.get_messages("record"):
        d = {d.name: d.value for d in m}
        if "timestamp" in d:
            recs.append({
                "time": d["timestamp"],                    # naive dt in device tz (assume UTC)
                "hr": d.get("heart_rate", np.nan),
                "power": d.get("power", np.nan),
            })

    df = pd.DataFrame(recs)
    if not df.empty:
        df = df.sort_values("time")

    # Strength-setit (jos on)
    sets = []
    for m in ff.get_messages("set"):
        d = {d.name: d.value for d in m}
        sets.append(d)
    df_sets = pd.DataFrame(sets)

    out_summary = None
    if not df.empty:
        # normalisoi aikavyöhyke näyttöä varten
        start_utc = pd.to_datetime(df["time"].iloc[0]).tz_localize("UTC")
        end_utc   = pd.to_datetime(df["time"].iloc[-1]).tz_localize("UTC")
        local = tz.gettz(tz_name)
        start_local = start_utc.astimezone(local)
        end_local   = end_utc.astimezone(local)
        dur_sec = int((end_utc - start_utc).total_seconds()) + 1
        dur_hr  = dur_sec/3600.0
        # tehdään 1 s taajuus NP:lle
        df = df.set_index(pd.to_datetime(df["time"]).dt.tz_localize("UTC")).sort_index()
        df = df.resample("1s").ffill()
        avg_hr = float(df["hr"].mean()) if df["hr"].notna().any() else np.nan
        max_hr = float(df["hr"].max()) if df["hr"].notna().any() else np.nan
        avg_p  = float(df["power"].mean()) if df["power"].notna().any() else np.nan
        max_p  = float(df["power"].max()) if df["power"].notna().any() else np.nan
        npw    = np_power(df["power"].fillna(0)) if df["power"].notna().any() else np.nan
        IF     = (npw/ftp) if np.isfinite(npw) and ftp>0 else np.nan
        TSS    = ((dur_hr * npw * IF)/ftp * 100) if np.all(np.isfinite([dur_hr,npw,IF])) and ftp>0 else np.nan
        TRIMP  = trimp_from_hr(df["hr"].ffill(), hr_rest=hr_rest, hr_max=hr_max) if df["hr"].notna().any() else 0.0

        out_summary = {
            "file": path,
            "sport": session_sport,
            "sub_sport": session_subsport,
            "date": start_local.date().isoformat(),
            "start_time": start_local.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": end_local.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_min": round(dur_sec/60.0, 1),
            "avg_hr": round(avg_hr,1) if np.isfinite(avg_hr) else "",
            "max_hr": int(max_hr) if np.isfinite(max_hr) else "",
            "avg_power_w": round(avg_p,1) if np.isfinite(avg_p) else "",
            "max_power_w": round(max_p,1) if np.isfinite(max_p) else "",
            "np_w": round(npw,1) if np.isfinite(npw) else "",
            "IF": round(IF,3) if np.isfinite(IF) else "",
            "TSS": round(TSS,1) if np.isfinite(TSS) else "",
            "TRIMP": round(TRIMP,1),
        }
    return out_summary, df_sets

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fit_files", nargs="+")
    ap.add_argument("--ftp", type=float, required=True)
    ap.add_argument("--hrrest", type=int, default=50)
    ap.add_argument("--hrmax", type=int, default=190)
    ap.add_argument("--tz", type=str, default="Europe/Helsinki")
    ap.add_argument("--dump-sets", action="store_true", help="Tallenna strength-sarjat CSV:ksi jos löytyy")
    ap.add_argument("--multisport", action="store_true", help="Process multisport activities by session")
    args = ap.parse_args()

    rows = []
    all_sets = []
    processed_sessions = set()  # Track processed sessions to avoid duplicates

    for f in args.fit_files:
        if args.multisport:
            # Check if this file contains multisport data
            results, df_sets = summarize_fit_sessions(f, ftp=args.ftp, hr_rest=args.hrrest, hr_max=args.hrmax, tz_name=args.tz)

            for result in results:
                if result:
                    # Create a unique key for this session based on sport, start time, and duration
                    session_key = (
                        result.get('sport'),
                        result.get('start_time'),
                        result.get('duration_min'),
                        result.get('avg_hr', ''),
                        result.get('avg_power_w', '')
                    )

                    if session_key not in processed_sessions:
                        processed_sessions.add(session_key)
                        rows.append(result)
                    else:
                        print(f"Skipping duplicate session: {result['sport']} at {result['start_time']}")
        else:
            # Original behavior
            s, df_sets = summarize_fit_original(f, ftp=args.ftp, hr_rest=args.hrrest, hr_max=args.hrmax, tz_name=args.tz)
            if s: rows.append(s)

        if args.dump_sets and df_sets is not None and not df_sets.empty:
            df_sets.to_csv(Path(f).with_suffix("").name + "_strength_sets.csv", index=False)
            all_sets.append(Path(f).with_suffix("").name + "_strength_sets.csv")

    if rows:
        out = pd.DataFrame(rows).sort_values(["date","start_time"])
        # Remove internal columns before saving
        columns_to_remove = [col for col in out.columns if col.startswith('_')]
        out_clean = out.drop(columns=columns_to_remove)
        out_clean.to_csv("workout_summary_from_fit.csv", index=False)
        print("\n[VALMIS] workout_summary_from_fit.csv luotu.")
        print(out_clean.to_string(index=False))
        if all_sets:
            print("[SETIT] Strength-setit tallennettu tiedostoihin:")
            for p in all_sets: print(" -", p)
    else:
        print("Ei record-dataa: tarkista FIT-tiedosto.")

if __name__ == "__main__":
    from pathlib import Path
    main()