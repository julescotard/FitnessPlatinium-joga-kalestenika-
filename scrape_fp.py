import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, time
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

from dateutil import parser as dtparser
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


WEEKDAY_MAP = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}


@dataclass(frozen=True)
class Event:
    uid: str
    start_utc: datetime
    end_utc: datetime
    title: str
    location: str
    source_url: str
    category: str  # "yoga" | "calisthenics"


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_hhmm(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


def _build_windows(cfg: Dict[str, Any]) -> Dict[int, List[Tuple[time, time]]]:
    out: Dict[int, List[Tuple[time, time]]] = {}
    for day_key, intervals in cfg["availability_windows"].items():
        wd = WEEKDAY_MAP[day_key.lower()]
        out[wd] = [(_parse_hhmm(a), _parse_hhmm(b)) for a, b in intervals]
    return out


def _in_windows(start_local: datetime, end_local: datetime, windows: Dict[int, List[Tuple[time, time]]]) -> bool:
    wd = start_local.weekday()
    if wd not in windows:
        return False
    for a, b in windows[wd]:
        if start_local.time() >= a and end_local.time() <= b:
            return True
    return False


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def _guess_club_name(page_text: str, hints: List[str], fallback: str) -> str:
    t = page_text or ""
    for h in hints:
        if h in t:
            return f"Fitness Platinium {h}"
    return fallback


def _iter_json_objects(obj: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _iter_json_objects(v)
    elif isinstance(obj, list):
        for x in obj:
            yield from _iter_json_objects(x)


def _extract_dt(d: Dict[str, Any], keys: List[str]) -> Optional[datetime]:
    for k in keys:
        if k in d and d[k]:
            try:
                return dtparser.parse(str(d[k]))
            except Exception:
                pass
    return None


def _extract_str(d: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _looks_like_event(d: Dict[str, Any]) -> bool:
    start = _extract_dt(d, ["start", "startDate", "startDateTime", "startsAt", "dateFrom", "from", "begin"])
    name = _extract_str(d, ["name", "title", "className", "activityName", "serviceName", "displayName"])
    return bool(start and name)


def _event_uid(seed: str) -> str:
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()


def _to_utc(dt_local: datetime, tz: ZoneInfo) -> datetime:
    if dt_local.tzinfo is None:
        dt_local = dt_local.replace(tzinfo=tz)
    return dt_local.astimezone(ZoneInfo("UTC"))


def _write_ics(path: str, cal_name: str, events: List[Event]) -> None:
    now_utc = datetime.utcnow().replace(microsecond=0)
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//fp-smartplatinium-ics//PL",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        f"X-WR-CALNAME:{cal_name}",
    ]
    for ev in sorted(events, key=lambda e: e.start_utc):
        dtstamp = now_utc.strftime("%Y%m%dT%H%M%SZ")
        dtstart = ev.start_utc.strftime("%Y%m%dT%H%M%SZ")
        dtend = ev.end_utc.strftime("%Y%m%dT%H%M%SZ")
        summary = ev.title.replace("\n", " ").replace("\r", " ")
        location = ev.location.replace("\n", " ").replace("\r", " ")
        desc = f"Źródło: {ev.source_url}".replace("\n", " ").replace("\r", " ")
        lines += [
            "BEGIN:VEVENT",
            f"UID:{ev.uid}",
            f"DTSTAMP:{dtstamp}",
            f"DTSTART:{dtstart}",
            f"DTEND:{dtend}",
            f"SUMMARY:{summary}",
            f"LOCATION:{location}",
            f"DESCRIPTION:{desc}",
            f"URL:{ev.source_url}",
            "END:VEVENT",
        ]
    lines.append("END:VCALENDAR")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--out", default="docs")
    args = ap.parse_args()

    cfg = _load_config(args.config)
    tz = ZoneInfo(cfg.get("timezone", "Europe/Warsaw"))

    # DST-proof: uruchamiamy workflow 2x (UTC 05:15 i 06:15),
    # ale wykonujemy scraping tylko, gdy lokalnie jest godzina z configu.
    only_hour = cfg.get("run_only_at_local_hour", None)
    force_run = os.getenv("FORCE_RUN", "0") == "1"
    if only_hour is not None and not force_run:
        now_local = datetime.now(tz=tz)
        if int(now_local.hour) != int(only_hour):
            print(f"[SKIP] Local time is {now_local.isoformat(timespec='minutes')} (want hour={only_hour}).")
            return

    days_ahead = int(args.days or cfg.get("days_ahead", 35))
    windows = _build_windows(cfg)
    keywords = {k: [_norm(x) for x in v] for k, v in cfg["keywords"].items()}
    hints = cfg.get("club_name_hints", [])

    os.makedirs(args.out, exist_ok=True)

    prev_path = os.path.join(args.out, "availability_summary.json")
    prev_summary = {}
    if os.path.exists(prev_path):
        try:
            with open(prev_path, "r", encoding="utf-8") as f:
                prev_summary = json.load(f)
        except Exception:
            prev_summary = {}

    start_cutoff = datetime.now(tz=tz).replace(hour=0, minute=0, second=0, microsecond=0)
    end_cutoff = start_cutoff + timedelta(days=days_ahead)

    all_events: List[Event] = []
    errors: List[str] = []
    clubs_seen: Dict[str, Dict[str, object]] = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(locale="pl-PL", timezone_id=str(tz.key))
        page = context.new_page()

        for cal_url in cfg["club_calendars"]:
            club_fallback = f"SmartPlatinium Classes URL: {cal_url.split('#')[-1]}"
            club_name = club_fallback
            captured_json: List[Any] = []

            def on_response(resp):
                try:
                    ct = resp.headers.get("content-type", "")
                    if "application/json" in ct:
                        data = resp.json()
                        captured_json.append(data)
                except Exception:
                    pass

            page.on("response", on_response)

            try:
                list_url = cal_url.replace("/Calendar", "/List")
                page.goto(list_url, wait_until="domcontentloaded", timeout=60000)

                for txt in ["Akceptuj", "Zgadzam się", "Rozumiem", "Accept"]:
                    try:
                        btn = page.get_by_role("button", name=re.compile(txt, re.I))
                        if btn.count() > 0:
                            btn.first.click(timeout=1500)
                            break
                    except Exception:
                        pass

                page.wait_for_timeout(3500)

                # scroll: dociągnij kolejne dni (więcej, bo days_ahead=35)
                for _ in range(14):
                    page.mouse.wheel(0, 2000)
                    page.wait_for_timeout(1100)

                body_text = ""
                try:
                    body_text = page.inner_text("body")
                except Exception:
                    pass
                club_name = _guess_club_name(body_text, hints, club_fallback)

                raw_candidates: List[Dict[str, Any]] = []
                for blob in captured_json:
                    for obj in _iter_json_objects(blob):
                        if _looks_like_event(obj):
                            raw_candidates.append(obj)

                yoga_any = False
                cal_any = False

                for d in raw_candidates:
                    name = _extract_str(d, ["name", "title", "className", "activityName", "serviceName", "displayName"]) or ""
                    name_n = _norm(name)

                    category = None
                    if any(k in name_n for k in keywords["yoga"]):
                        category = "yoga"
                        yoga_any = True
                    elif any(k in name_n for k in keywords["calisthenics"]):
                        category = "calisthenics"
                        cal_any = True
                    else:
                        continue

                    start_dt = _extract_dt(d, ["start", "startDate", "startDateTime", "startsAt", "dateFrom", "from", "begin"])
                    end_dt = _extract_dt(d, ["end", "endDate", "endDateTime", "endsAt", "dateTo", "to", "finish"])
                    if start_dt and not end_dt:
                        end_dt = start_dt + timedelta(minutes=60)

                    if not start_dt or not end_dt:
                        continue

                    if start_dt.tzinfo is None:
                        start_dt = start_dt.replace(tzinfo=tz)
                    else:
                        start_dt = start_dt.astimezone(tz)

                    if end_dt.tzinfo is None:
                        end_dt = end_dt.replace(tzinfo=tz)
                    else:
                        end_dt = end_dt.astimezone(tz)

                    if not (start_cutoff <= start_dt < end_cutoff):
                        continue

                    uid_seed = _extract_str(d, ["id", "classId", "eventId", "slotId", "bookingId"]) or f"{club_name}|{name}|{start_dt.isoformat()}|{end_dt.isoformat()}"
                    uid = _event_uid(uid_seed)

                    all_events.append(
                        Event(
                            uid=uid,
                            start_utc=_to_utc(start_dt, tz),
                            end_utc=_to_utc(end_dt, tz),
                            title=f"{'Joga' if category=='yoga' else 'Kalistenika'} — {name}".strip(),
                            location=club_name,
                            source_url=list_url,
                            category=category,
                        )
                    )

                clubs_seen[club_name] = {"yoga": yoga_any, "calisthenics": cal_any, "ok": True, "url": cal_url}

            except PlaywrightTimeoutError:
                errors.append(f"TIMEOUT: {cal_url}")
                clubs_seen[club_name] = {"yoga": False, "calisthenics": False, "ok": False, "url": cal_url}
            except Exception as e:
                errors.append(f"ERROR: {cal_url} :: {type(e).__name__}: {e}")
                clubs_seen[club_name] = {"yoga": False, "calisthenics": False, "ok": False, "url": cal_url}
            finally:
                try:
                    page.remove_listener("response", on_response)
                except Exception:
                    pass

        context.close()
        browser.close()

    # dedupe
    all_events = list({ev.uid: ev for ev in all_events}.values())

    # filtr do Twoich okien (tylko do plików "moje okna")
    filtered_events: List[Event] = []
    for ev in all_events:
        start_local = ev.start_utc.astimezone(tz)
        end_local = ev.end_utc.astimezone(tz)
        if _in_windows(start_local, end_local, windows):
            filtered_events.append(ev)

    yoga_events = [e for e in all_events if e.category == "yoga"]
    cal_events = [e for e in all_events if e.category == "calisthenics"]
    yoga_filtered = [e for e in filtered_events if e.category == "yoga"]
    cal_filtered = [e for e in filtered_events if e.category == "calisthenics"]

    # ICS: pełne (WSZYSTKIE DNI/GODZINY) + osobno wariant "moje okna"
    _write_ics(os.path.join(args.out, "fp-yoga-kalistenika.ics"), "FP: Joga + Kalistenika", all_events)
    _write_ics(os.path.join(args.out, "fp-yoga-kalistenika-moje-okna.ics"), "FP: Joga + Kalistenika (moje okna)", filtered_events)
    _write_ics(os.path.join(args.out, "fp-yoga.ics"), "FP: Joga", yoga_events)
    _write_ics(os.path.join(args.out, "fp-kalistenika.ics"), "FP: Kalistenika", cal_events)
    _write_ics(os.path.join(args.out, "fp-yoga-moje-okna.ics"), "FP: Joga (moje okna)", yoga_filtered)
    _write_ics(os.path.join(args.out, "fp-kalistenika-moje-okna.ics"), "FP: Kalistenika (moje okna)", cal_filtered)

    yoga_clubs = sorted([c for c, v in clubs_seen.items() if v.get("ok") and v.get("yoga")])
    cal_clubs = sorted([c for c, v in clubs_seen.items() if v.get("ok") and v.get("calisthenics")])

    now_iso = datetime.now(tz=tz).isoformat(timespec="seconds")
    summary = {
        "generated_at": now_iso,
        "days_ahead": days_ahead,
        "clubs_seen": clubs_seen,
        "yoga_clubs": yoga_clubs,
        "calisthenics_clubs": cal_clubs,
        "errors": errors,
    }

    with open(os.path.join(args.out, "availability_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    prev_yoga = set(prev_summary.get("yoga_clubs", [])) if isinstance(prev_summary, dict) else set()
    prev_cal = set(prev_summary.get("calisthenics_clubs", [])) if isinstance(prev_summary, dict) else set()
    cur_union = set(yoga_clubs) | set(cal_clubs)
    prev_union = prev_yoga | prev_cal

    added = sorted(cur_union - prev_union)
    removed = sorted(prev_union - cur_union)

    changes = {"date": now_iso, "added": added, "removed": removed}
    with open(os.path.join(args.out, "changes.json"), "w", encoding="utf-8") as f:
        json.dump(changes, f, ensure_ascii=False, indent=2)

    md_lines = [
        "# Zmiany dostępności (joga/kalistenika)",
        f"- wygenerowano: **{now_iso}**",
        f"- zakres: **{days_ahead} dni do przodu**",
        "",
        "## Dodane lokalizacje",
        *(["- (brak)"] if not added else [f"- {x}" for x in added]),
        "",
        "## Usunięte lokalizacje",
        *(["- (brak)"] if not removed else [f"- {x}" for x in removed]),
        "",
        "## Aktualnie (kluby z jogą)",
        *(["- (brak)"] if not yoga_clubs else [f"- {x}" for x in yoga_clubs]),
        "",
        "## Aktualnie (kluby z kalisteniką)",
        *(["- (brak)"] if not cal_clubs else [f"- {x}" for x in cal_clubs]),
        "",
    ]
    if errors:
        md_lines += ["## Błędy odczytu (nie liczone jako zmiany)", *[f"- {e}" for e in errors], ""]

    with open(os.path.join(args.out, "changes.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    index_html = f"""<!doctype html>
<html lang="pl">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>FP – Joga/Kalistenika (ICS)</title>
  <style>body{{font-family:system-ui,Arial,sans-serif;max-width:900px;margin:40px auto;padding:0 16px;}}code{{background:#f4f4f4;padding:2px 6px;border-radius:6px;}}</style>
</head>
<body>
  <h1>FP – Joga/Kalistenika (ICS)</h1>
  <p>Wygenerowano: <strong>{now_iso}</strong></p>

  <h2>Subskrypcje ICS</h2>
  <ul>
    <li><a href="fp-yoga-kalistenika.ics">fp-yoga-kalistenika.ics</a> (WSZYSTKIE zajęcia)</li>
    <li><a href="fp-yoga-kalistenika-moje-okna.ics">fp-yoga-kalistenika-moje-okna.ics</a> (tylko Twoje okna)</li>
    <li><a href="fp-yoga.ics">fp-yoga.ics</a> (wszystkie)</li>
    <li><a href="fp-kalistenika.ics">fp-kalistenika.ics</a> (wszystkie)</li>
    <li><a href="fp-yoga-moje-okna.ics">fp-yoga-moje-okna.ics</a></li>
    <li><a href="fp-kalistenika-moje-okna.ics">fp-kalistenika-moje-okna.ics</a></li>
  </ul>

  <h2>Zmiany lokalizacji</h2>
  <ul>
    <li><a href="changes.md">changes.md</a></li>
    <li><a href="changes.json">changes.json</a></li>
    <li><a href="availability_summary.json">availability_summary.json</a></li>
  </ul>
</body>
</html>
"""
    with open(os.path.join(args.out, "index.html"), "w", encoding="utf-8") as f:
        f.write(index_html)


if __name__ == "__main__":
    main()
