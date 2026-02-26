import argparse
import copy
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, time, date
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from zoneinfo import ZoneInfo
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

from dateutil import parser as dtparser
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


WEEKDAY_MAP = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}

DATE_KEYS = ["date", "Date", "day", "Day", "classDate", "ClassDate", "dateFrom", "DateFrom", "startDate", "StartDate", "calendarDate", "CalendarDate", "selectedDate", "SelectedDate"]
TIME_START_KEYS = ["startTime", "StartTime", "fromTime", "FromTime", "timeFrom", "TimeFrom", "hourFrom", "HourFrom", "startHour", "StartHour"]
TIME_END_KEYS = ["endTime", "EndTime", "toTime", "ToTime", "timeTo", "TimeTo", "hourTo", "HourTo", "endHour", "EndHour"]

DT_KEYS_START = [
    "start", "Start", "startDate", "StartDate", "startDateTime", "StartDateTime",
    "startsAt", "StartsAt", "dateFrom", "DateFrom", "from", "From", "begin", "Begin",
    "startUtc", "StartUtc", "startUTC", "StartUTC",
    "startDateUtc", "StartDateUtc", "startDateTimeUtc", "StartDateTimeUtc", "startDateTimeUTC", "StartDateTimeUTC",
    "StartTime"
]
DT_KEYS_END = [
    "end", "End", "endDate", "EndDate", "endDateTime", "EndDateTime",
    "endsAt", "EndsAt", "dateTo", "DateTo", "to", "To", "finish", "Finish",
    "endUtc", "EndUtc", "endUTC", "EndUTC",
    "endDateUtc", "EndDateUtc", "endDateTimeUtc", "EndDateTimeUtc", "endDateTimeUTC", "EndDateTimeUTC",
    "EndTime"
]

NAME_KEYS_PRIMARY = [
    "serviceName", "ServiceName",
    "activityName", "ActivityName",
    "className", "ClassName",
    "displayName", "DisplayName",
    "title", "Title",
    "name", "Name",
]


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


def _parse_hhmm(s: Any) -> Optional[time]:
    try:
        s = str(s).strip()
        parts = s.split(":")
        if len(parts) >= 2:
            return time(int(parts[0]), int(parts[1]))
    except Exception:
        pass
    return None


def _build_windows(cfg: Dict[str, Any]) -> Dict[int, List[Tuple[time, time]]]:
    out: Dict[int, List[Tuple[time, time]]] = {}
    for day_key, intervals in cfg["availability_windows"].items():
        wd = WEEKDAY_MAP[day_key.lower()]
        tmp = []
        for a, b in intervals:
            ta = _parse_hhmm(a)
            tb = _parse_hhmm(b)
            if ta and tb:
                tmp.append((ta, tb))
        out[wd] = tmp
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


def _parse_dotnet_date(s: str) -> Optional[datetime]:
    m = re.search(r"Date\((\-?\d+)\)", s)
    if not m:
        return None
    try:
        ms = int(m.group(1))
        return datetime.fromtimestamp(ms / 1000, tz=ZoneInfo("UTC"))
    except Exception:
        return None


def _parse_epoch(value: Any) -> Optional[datetime]:
    try:
        if isinstance(value, (int, float)):
            n = float(value)
            if n > 1e12:  # ms
                return datetime.fromtimestamp(n / 1000, tz=ZoneInfo("UTC"))
            if n > 1e9:  # sec
                return datetime.fromtimestamp(n, tz=ZoneInfo("UTC"))
        if isinstance(value, str) and value.strip().isdigit():
            return _parse_epoch(int(value))
    except Exception:
        pass
    return None


def _extract_dt_value(v: Any) -> Optional[datetime]:
    if v is None or v == "":
        return None
    if isinstance(v, datetime):
        return v
    if isinstance(v, (int, float)) or (isinstance(v, str) and v.strip().isdigit()):
        return _parse_epoch(v)
    if isinstance(v, str):
        dn = _parse_dotnet_date(v)
        if dn:
            return dn
        try:
            return dtparser.parse(v)
        except Exception:
            return None
    return None


def _extract_dt(d: Dict[str, Any], keys: List[str]) -> Optional[datetime]:
    for k in keys:
        if k in d:
            dt = _extract_dt_value(d.get(k))
            if dt:
                return dt
    return None


def _extract_date(d: Dict[str, Any]) -> Optional[date]:
    for k in DATE_KEYS:
        if k in d and d.get(k) not in (None, ""):
            v = d.get(k)
            if isinstance(v, str) and "Date(" in v:
                dt = _parse_dotnet_date(v)
                if dt:
                    return dt.date()
            try:
                dt = dtparser.parse(str(v))
                return dt.date()
            except Exception:
                pass
    return None


def _extract_time(d: Dict[str, Any], keys: List[str]) -> Optional[time]:
    for k in keys:
        if k in d and d.get(k) not in (None, ""):
            t = _parse_hhmm(d.get(k))
            if t:
                return t
            try:
                v = d.get(k)
                if isinstance(v, (int, float)):
                    mins = int(v)
                    if 0 <= mins < 24 * 60:
                        return time(mins // 60, mins % 60)
            except Exception:
                pass
    return None


def _extract_name(d: Dict[str, Any]) -> Optional[str]:
    for k in NAME_KEYS_PRIMARY:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    for k in ["service", "Service", "activity", "Activity", "classType", "ClassType"]:
        v = d.get(k)
        if isinstance(v, dict):
            for kk in ["name", "Name", "title", "Title", "displayName", "DisplayName"]:
                vv = v.get(kk)
                if isinstance(vv, str) and vv.strip():
                    return vv.strip()
    return None


def _collect_all_strings(d: Dict[str, Any], max_items: int = 120) -> str:
    out: List[str] = []
    def rec(x: Any):
        if len(out) >= max_items:
            return
        if isinstance(x, dict):
            for v in x.values():
                rec(v)
        elif isinstance(x, list):
            for v in x[:16]:
                rec(v)
        elif isinstance(x, str):
            if x.strip():
                out.append(x.strip())
    rec(d)
    return " ".join(out)


def _parse_duration_to_minutes(value: Any) -> Optional[int]:
    if not value:
        return None
    s = str(value).strip()
    if not s:
        return None
    s = s.replace("-", "")
    m = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", s)
    if m:
        h = int(m.group(1) or 0)
        mi = int(m.group(2) or 0)
        sec = int(m.group(3) or 0)
        return h * 60 + mi + (1 if sec >= 30 else 0)
    return None


def _looks_like_event(d: Dict[str, Any]) -> bool:
    name = _extract_name(d)
    if not name:
        return False
    if _extract_dt(d, DT_KEYS_START):
        return True
    if _extract_date(d) and _extract_time(d, TIME_START_KEYS):
        return True
    return False


def _event_uid(seed: str) -> str:
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()


def _to_utc_from_local(dt_local: datetime, tz: ZoneInfo) -> datetime:
    if dt_local.tzinfo is None:
        dt_local = dt_local.replace(tzinfo=tz)
    return dt_local.astimezone(ZoneInfo("UTC"))


def _ics_escape(value: str) -> str:
    v = (value or "")
    v = v.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,")
    v = v.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")
    return v


def _fold_ical_line(line: str, limit: int = 75) -> List[str]:
    b = line.encode("utf-8")
    if len(b) <= limit:
        return [line]
    out: List[str] = []
    cur_bytes = bytearray()
    cur_chars: List[str] = []

    def flush(prefix_space: bool = False):
        if cur_chars:
            out.append((" " if prefix_space else "") + "".join(cur_chars))

    first = True
    for ch in line:
        chb = ch.encode("utf-8")
        if len(cur_bytes) + len(chb) > limit:
            flush(prefix_space=not first)
            first = False
            cur_bytes = bytearray()
            cur_chars = []
        cur_bytes.extend(chb)
        cur_chars.append(ch)
    flush(prefix_space=not first)
    return out


def _write_ics(path: str, cal_name: str, events: List[Event], tz_name: str) -> None:
    now_utc = datetime.utcnow().replace(microsecond=0)
    lines: List[str] = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//fp-smartplatinium-ics//PL",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        f"X-WR-CALNAME:{_ics_escape(cal_name)}",
        f"X-WR-TIMEZONE:{tz_name}",
        "REFRESH-INTERVAL;VALUE=DURATION:PT1H",
        "X-PUBLISHED-TTL:PT1H",
    ]
    for ev in sorted(events, key=lambda e: e.start_utc):
        dtstamp = now_utc.strftime("%Y%m%dT%H%M%SZ")
        dtstart = ev.start_utc.strftime("%Y%m%dT%H%M%SZ")
        dtend = ev.end_utc.strftime("%Y%m%dT%H%M%SZ")
        uid = f"{ev.uid}@fp-smartplatinium"
        vevent = [
            "BEGIN:VEVENT",
            f"UID:{uid}",
            f"DTSTAMP:{dtstamp}",
            f"DTSTART:{dtstart}",
            f"DTEND:{dtend}",
            f"SUMMARY:{_ics_escape(ev.title)}",
            f"LOCATION:{_ics_escape(ev.location)}",
            f"DESCRIPTION:{_ics_escape('Źródło: ' + ev.source_url)}",
            f"URL:{ev.source_url}",
            "END:VEVENT",
        ]
        for l in vevent:
            lines.extend(_fold_ical_line(l))
    lines.append("END:VCALENDAR")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\r\n".join(lines) + "\r\n")


def _try_parse_json_text(txt: str) -> Optional[Any]:
    if not txt:
        return None
    t = txt.strip()
    if not t:
        return None
    idx = None
    for ch in ("{", "["):
        p = t.find(ch)
        if p != -1:
            idx = p if idx is None else min(idx, p)
    if idx is None:
        return None
    candidate = t[idx:]
    last_obj = candidate.rfind("}")
    last_arr = candidate.rfind("]")
    last = max(last_obj, last_arr)
    if last != -1:
        candidate = candidate[: last + 1]
    try:
        return json.loads(candidate)
    except Exception:
        return None


def _event_key(ev: Event, tz: ZoneInfo) -> str:
    st = ev.start_utc.astimezone(tz).strftime("%Y-%m-%d %H:%M")
    en = ev.end_utc.astimezone(tz).strftime("%H:%M")
    return f"{st}-{en} | {ev.category} | {ev.location} | {ev.title}"


def _filter_headers(h: Dict[str, str]) -> Dict[str, str]:
    # keep only safe headers; cookies handled by context.request
    keep = {}
    for k, v in (h or {}).items():
        kl = k.lower()
        if kl in {"content-type", "x-requested-with", "requestverificationtoken"}:
            keep[k] = v
    return keep


def _set_date_in_anything(obj: Any, new_date: date) -> Any:
    """Replace any YYYY-MM-DD in string values inside obj."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out[k] = _set_date_in_anything(v, new_date)
        return out
    if isinstance(obj, list):
        return [_set_date_in_anything(v, new_date) for v in obj]
    if isinstance(obj, str):
        m = re.match(r"^(\d{4}-\d{2}-\d{2})(T.*)?$", obj.strip())
        if m:
            return new_date.isoformat() + (m.group(2) or "")
        return obj
    return obj


def _replace_date_in_query(url: str, new_date: date) -> str:
    """Replace any query param that looks like a date with new_date."""
    p = urlparse(url)
    qs = parse_qs(p.query, keep_blank_values=True)
    changed = False
    for k, vals in list(qs.items()):
        if not vals:
            continue
        v0 = vals[0]
        if isinstance(v0, str):
            if re.match(r"^\d{4}-\d{2}-\d{2}$", v0) or re.match(r"^\d{4}-\d{2}-\d{2}T", v0):
                qs[k] = [new_date.isoformat()]
                changed = True
    if not changed:
        return url
    new_query = urlencode(qs, doseq=True)
    return urlunparse((p.scheme, p.netloc, p.path, p.params, new_query, p.fragment))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--out", default="docs")
    args = ap.parse_args()

    cfg = _load_config(args.config)
    tz_name = cfg.get("timezone", "Europe/Warsaw")
    tz = ZoneInfo(tz_name)

    only_hour = cfg.get("run_only_at_local_hour", None)
    force_run = os.getenv("FORCE_RUN", "0") == "1"
    if only_hour is not None and not force_run:
        now_local = datetime.now(tz=tz)
        if int(now_local.hour) != int(only_hour):
            print(f"[SKIP] Local time is {now_local.isoformat(timespec='minutes')} (want hour={only_hour}).")
            return

    days_ahead = int(args.days or cfg.get("days_ahead", 42))
    windows = _build_windows(cfg)
    keywords = {k: [_norm(x) for x in v] for k, v in cfg["keywords"].items()}
    hints = cfg.get("club_name_hints", [])

    forced = cfg.get("forced_category_by_timetableId", {})  # optional: {"36":"calisthenics","22":"yoga","296":"yoga"}

    os.makedirs(args.out, exist_ok=True)

    prev_summary = {}
    prev_summary_path = os.path.join(args.out, "availability_summary.json")
    if os.path.exists(prev_summary_path):
        try:
            with open(prev_summary_path, "r", encoding="utf-8") as f:
                prev_summary = json.load(f)
        except Exception:
            prev_summary = {}

    prev_event_keys: Set[str] = set()
    prev_events_path = os.path.join(args.out, "events_index.json")
    if os.path.exists(prev_events_path):
        try:
            with open(prev_events_path, "r", encoding="utf-8") as f:
                prev = json.load(f)
                if isinstance(prev, list):
                    prev_event_keys = set(str(x) for x in prev)
        except Exception:
            prev_event_keys = set()

    start_cutoff = datetime.now(tz=tz).replace(hour=0, minute=0, second=0, microsecond=0)
    end_cutoff = start_cutoff + timedelta(days=days_ahead)

    all_events: List[Event] = []
    errors: List[str] = []
    clubs_seen: Dict[str, Dict[str, object]] = {}
    debug_calls: Dict[str, List[Dict[str, object]]] = {}
    sample_payloads: Dict[str, Any] = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(locale="pl-PL", timezone_id=str(tz.key))
        page = context.new_page()

        for cal_url in cfg["club_calendars"]:
            club_fallback = f"SmartPlatinium Classes URL: {cal_url.split('#')[-1]}"
            club_name = club_fallback
            captured_blobs: List[Any] = []
            calls: List[Dict[str, object]] = []

            # capture first DailyClasses request template
            template = {"url": None, "method": None, "headers": None, "post_data": None, "post_is_json": False}
            sample_payloads[cal_url] = {"template": None, "daily_samples": []}

            def on_request(req):
                try:
                    if "Classes/ClassCalendar/DailyClasses" not in req.url:
                        return
                    if template["url"] is not None:
                        return
                    template["url"] = req.url
                    template["method"] = req.method
                    template["headers"] = req.headers
                    pd = req.post_data or ""
                    template["post_data"] = pd
                    if pd.strip().startswith(("{","[")):
                        try:
                            json.loads(pd)
                            template["post_is_json"] = True
                        except Exception:
                            template["post_is_json"] = False
                except Exception:
                    pass

            def on_response(resp):
                try:
                    ct = (resp.headers.get("content-type", "") or "").lower()
                    calls.append({"url": resp.url, "status": resp.status, "content_type": ct[:80]})

                    obj = None
                    if "json" in ct:
                        try:
                            obj = resp.json()
                        except Exception:
                            obj = None
                    if obj is None:
                        try:
                            txt = resp.text()
                        except Exception:
                            txt = None
                        if txt:
                            obj = _try_parse_json_text(txt)

                    if obj is None:
                        return

                    captured_blobs.append(obj)

                    if "DailyClasses" in resp.url and len(sample_payloads[cal_url]["daily_samples"]) < 2:
                        cal = obj.get("CalendarData") if isinstance(obj, dict) else None
                        sample_payloads[cal_url]["daily_samples"].append({
                            "url": resp.url,
                            "has_data": bool(isinstance(cal, list) and len(cal) > 0),
                            "keys": list(obj.keys())[:30] if isinstance(obj, dict) else None,
                        })
                except Exception:
                    pass

            page.on("request", on_request)
            page.on("response", on_response)

            try:
                # Use Calendar view (more reliable to trigger API)
                page.goto(cal_url, wait_until="domcontentloaded", timeout=60000)

                # cookies
                for txt in ["Akceptuj", "Zgadzam się", "Rozumiem", "Accept"]:
                    try:
                        btn = page.get_by_role("button", name=re.compile(txt, re.I))
                        if btn.count() > 0:
                            btn.first.click(timeout=1500)
                            break
                    except Exception:
                        pass

                page.wait_for_timeout(3500)

                # small scroll to trigger loads
                for _ in range(6):
                    page.mouse.wheel(0, 2200)
                    page.wait_for_timeout(600)

                # Try to read club name
                try:
                    club_name = _guess_club_name(page.inner_text("body"), hints, club_fallback)
                except Exception:
                    club_name = club_fallback

                # API iteration: if template captured, request each day
                if template["url"] and template["method"]:
                    sample_payloads[cal_url]["template"] = {
                        "url": template["url"],
                        "method": template["method"],
                        "headers": _filter_headers(template["headers"] or {}),
                        "post_is_json": template["post_is_json"],
                        "post_len": len(template["post_data"] or ""),
                    }

                    method = str(template["method"]).upper()
                    headers = _filter_headers(template["headers"] or {})
                    endpoint = template["url"]
                    raw_post = template["post_data"] or ""

                    d = start_cutoff.date()
                    end_d = (end_cutoff - timedelta(days=1)).date()

                    while d <= end_d:
                        try:
                            if method == "GET":
                                u = _replace_date_in_query(endpoint, d)
                                r = context.request.get(u, headers=headers, timeout=60000)
                            else:
                                if template["post_is_json"]:
                                    payload = json.loads(raw_post) if raw_post.strip().startswith(("{","[")) else {}
                                    payload = _set_date_in_anything(payload, d)
                                    r = context.request.post(endpoint, headers=headers, data=json.dumps(payload), timeout=60000)
                                else:
                                    # form-like data: replace YYYY-MM-DD patterns
                                    pd = re.sub(r"\d{4}-\d{2}-\d{2}", d.isoformat(), raw_post)
                                    r = context.request.post(endpoint, headers=headers, data=pd, timeout=60000)
                            if r.ok:
                                captured_blobs.append(r.json())
                        except Exception:
                            pass
                        d = d + timedelta(days=1)

                yoga_any = False
                cal_any = False
                events_here = 0

                # determine timetableId to allow forced categories
                tt = None
                m = re.search(r"timeTableId=(\d+)", cal_url)
                if m:
                    tt = m.group(1)

                for blob in captured_blobs:
                    for obj in _iter_json_objects(blob):
                        if not _looks_like_event(obj):
                            continue

                        name = _extract_name(obj) or ""
                        hay = _norm(name + " " + _collect_all_strings(obj))

                        category = None
                        if tt and str(tt) in forced:
                            category = forced[str(tt)]
                        else:
                            if any(k in hay for k in keywords["yoga"]):
                                category = "yoga"
                            elif any(k in hay for k in keywords["calisthenics"]):
                                category = "calisthenics"
                            else:
                                continue

                        if category == "yoga":
                            yoga_any = True
                        if category == "calisthenics":
                            cal_any = True

                        start_dt = _extract_dt(obj, DT_KEYS_START)
                        end_dt = _extract_dt(obj, DT_KEYS_END)

                        if start_dt:
                            start_local = start_dt.astimezone(tz) if start_dt.tzinfo else start_dt.replace(tzinfo=tz)
                        else:
                            d0 = _extract_date(obj)
                            t0 = _extract_time(obj, TIME_START_KEYS)
                            if not (d0 and t0):
                                continue
                            start_local = datetime.combine(d0, t0).replace(tzinfo=tz)

                        if end_dt:
                            end_local = end_dt.astimezone(tz) if end_dt.tzinfo else end_dt.replace(tzinfo=tz)
                        else:
                            dur_min = _parse_duration_to_minutes(obj.get("Duration") or obj.get("duration"))
                            if dur_min and dur_min >= 10:
                                end_local = start_local + timedelta(minutes=dur_min)
                            else:
                                end_local = start_local + timedelta(minutes=60)

                        if not (start_cutoff <= start_local < end_cutoff):
                            continue

                        # IMPORTANT: UID includes start time to avoid collapsing recurring classes.
                        uid_seed = f"{club_name}|{tt or ''}|{start_local.isoformat()}|{name}"
                        uid = _event_uid(uid_seed)

                        all_events.append(Event(
                            uid=uid,
                            start_utc=_to_utc_from_local(start_local, tz),
                            end_utc=_to_utc_from_local(end_local, tz),
                                                                               trainer = (
    obj.get("TrainerName") or obj.get("trainerName") or
    obj.get("InstructorName") or obj.get("instructorName") or
    obj.get("Trainer") or obj.get("trainer") or ""
)
if isinstance(trainer, dict):
    trainer = trainer.get("Name") or trainer.get("name") or ""
if not isinstance(trainer, str):
    trainer = ""
trainer = trainer.strip()
start_hhmm = start_local.strftime("%H:%M")
kind = "Joga" if category == "yoga" else "Kalistenika"
who = trainer if trainer else "bez trenera"
title = f"{start_hhmm} {kind} | {club_name} | {who} | {name}",
                            location=club_name,
                            source_url=cal_url,
                            category=category,
                        ))
                        events_here += 1

                clubs_seen[f"{club_name} :: {tt or '?'}"] = {
                    "yoga": bool(yoga_any),
                    "calisthenics": bool(cal_any),
                    "ok": True,
                    "url": cal_url,
                    "blobs": len(captured_blobs),
                    "calls": len(calls),
                    "events_found": events_here,
                    "template_captured": bool(template["url"]),
                    "template_method": template["method"],
                }
                debug_calls[cal_url] = calls[:300]

            except PlaywrightTimeoutError:
                errors.append(f"TIMEOUT: {cal_url}")
            except Exception as e:
                errors.append(f"ERROR: {cal_url} :: {type(e).__name__}: {e}")
            finally:
                try:
                    page.remove_listener("request", on_request)
                    page.remove_listener("response", on_response)
                except Exception:
                    pass

        context.close()
        browser.close()

    # no global dedup needed; but keep safe unique by uid
    all_events = list({ev.uid: ev for ev in all_events}.values())

    filtered_events: List[Event] = []
    for ev in all_events:
        st = ev.start_utc.astimezone(tz)
        en = ev.end_utc.astimezone(tz)
        if _in_windows(st, en, windows):
            filtered_events.append(ev)

    yoga_events = [e for e in all_events if e.category == "yoga"]
    cal_events = [e for e in all_events if e.category == "calisthenics"]
    yoga_filtered = [e for e in filtered_events if e.category == "yoga"]
    cal_filtered = [e for e in filtered_events if e.category == "calisthenics"]

    _write_ics(os.path.join(args.out, "fp-yoga-kalistenika.ics"), "FP: Joga + Kalistenika", all_events, tz_name)
    _write_ics(os.path.join(args.out, "fp-yoga-kalistenika-moje-okna.ics"), "FP: Joga + Kalistenika (moje okna)", filtered_events, tz_name)
    _write_ics(os.path.join(args.out, "fp-yoga.ics"), "FP: Joga", yoga_events, tz_name)
    _write_ics(os.path.join(args.out, "fp-kalistenika.ics"), "FP: Kalistenika", cal_events, tz_name)
    _write_ics(os.path.join(args.out, "fp-yoga-moje-okna.ics"), "FP: Joga (moje okna)", yoga_filtered, tz_name)
    _write_ics(os.path.join(args.out, "fp-kalistenika-moje-okna.ics"), "FP: Kalistenika (moje okna)", cal_filtered, tz_name)

    current_event_keys = sorted({_event_key(ev, tz) for ev in all_events})
    with open(os.path.join(args.out, "events_index.json"), "w", encoding="utf-8") as f:
        json.dump(current_event_keys, f, ensure_ascii=False, indent=2)

    cur_set = set(current_event_keys)
    added_events = sorted(cur_set - prev_event_keys)
    removed_events = sorted(prev_event_keys - cur_set)
    with open(os.path.join(args.out, "events_changes.json"), "w", encoding="utf-8") as f:
        json.dump({"date": datetime.now(tz=tz).isoformat(timespec="seconds"), "added": added_events, "removed": removed_events}, f, ensure_ascii=False, indent=2)

    yoga_clubs = sorted([c for c, v in clubs_seen.items() if v.get("ok") and v.get("yoga")])
    cal_clubs = sorted([c for c, v in clubs_seen.items() if v.get("ok") and v.get("calisthenics")])
    prev_union = set(prev_summary.get("yoga_clubs", [])) | set(prev_summary.get("calisthenics_clubs", []))
    cur_union = set(yoga_clubs) | set(cal_clubs)

    now_iso = datetime.now(tz=tz).isoformat(timespec="seconds")

    with open(os.path.join(args.out, "changes.json"), "w", encoding="utf-8") as f:
        json.dump({"date": now_iso, "added": sorted(cur_union - prev_union), "removed": sorted(prev_union - cur_union)}, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.out, "availability_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": now_iso,
            "days_ahead": days_ahead,
            "event_counts": {"all": len(all_events), "yoga": len(yoga_events), "calisthenics": len(cal_events), "my_windows": len(filtered_events)},
            "clubs_seen": clubs_seen,
            "yoga_clubs": yoga_clubs,
            "calisthenics_clubs": cal_clubs,
            "errors": errors,
        }, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.out, "debug_calls.json"), "w", encoding="utf-8") as f:
        json.dump(debug_calls, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.out, "sample_payloads.json"), "w", encoding="utf-8") as f:
        json.dump(sample_payloads, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.out, "changes.md"), "w", encoding="utf-8") as f:
        f.write(
            "# Zmiany (joga/kalistenika)\n"
            f"- wygenerowano: **{now_iso}**\n"
            f"- zakres: **{days_ahead} dni do przodu**\n"
            f"- terminy: all={len(all_events)}, yoga={len(yoga_events)}, calisthenics={len(cal_events)}, moje_okna={len(filtered_events)}\n"
        )

    index_html = f"""<!doctype html>
<html lang="pl">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>FP – Joga/Kalistenika (ICS)</title>
  <style>
    body{{font-family:system-ui,Arial,sans-serif;max-width:900px;margin:40px auto;padding:0 16px;}}
    code{{background:#f4f4f4;padding:2px 6px;border-radius:6px;}}
  </style>
</head>
<body>
  <h1>FP – Joga/Kalistenika (ICS)</h1>
  <p>Wygenerowano: <strong>{now_iso}</strong></p>
  <p>Terminy: <code>all={len(all_events)} | yoga={len(yoga_events)} | calisthenics={len(cal_events)} | moje_okna={len(filtered_events)}</code></p>

  <h2>Subskrypcje</h2>
  <ul>
    <li><a href="fp-yoga-kalistenika.ics">Joga+Kalistenika (wszystko)</a></li>
    <li><a href="fp-yoga-kalistenika-moje-okna.ics">Joga+Kalistenika (moje okna)</a></li>
    <li><a href="fp-yoga.ics">Joga (wszystko)</a></li>
    <li><a href="fp-kalistenika.ics">Kalistenika (wszystko)</a></li>
    <li><a href="fp-yoga-moje-okna.ics">Joga (moje okna)</a></li>
    <li><a href="fp-kalistenika-moje-okna.ics">Kalistenika (moje okna)</a></li>
  </ul>

  <h2>Diagnostyka</h2>
  <ul>
    <li><a href="availability_summary.json">availability_summary.json</a></li>
    <li><a href="changes.md">changes.md</a></li>
    <li><a href="changes.json">changes.json</a></li>
    <li><a href="events_changes.json">events_changes.json</a></li>
    <li><a href="debug_calls.json">debug_calls.json</a></li>
    <li><a href="sample_payloads.json">sample_payloads.json</a></li>
  </ul>
</body>
</html>
"""
    with open(os.path.join(args.out, "index.html"), "w", encoding="utf-8") as f:
        f.write(index_html)


if __name__ == "__main__":
    main()
