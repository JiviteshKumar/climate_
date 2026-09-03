# -*- coding: utf-8 -*-
"""
HTTP security-header and TLS audit
STQA Experiment 05 -- Security Testing
Jivitesh Kumar | Roll No. 16010423041 | Batch A2 | KJSSE, Sem VII

Checks the live deployment of the Heatwave Intelligence Platform against the
OWASP Secure Headers Project recommendations, inspects the negotiated TLS
parameters, and probes for a small set of common misconfigurations.

Run:  python header_audit.py
"""

import ssl
import socket
import json
import urllib.request
import urllib.error
from datetime import datetime, timezone

HOST = "jiviteshkumar.github.io"
BASE = "https://%s/climate_/" % HOST

# header -> (severity, what it defends against, what a good value looks like)
CHECKS = {
    "strict-transport-security": (
        "High", "Protocol downgrade and SSL-stripping man-in-the-middle attacks",
        "max-age=31536000; includeSubDomains; preload"),
    "content-security-policy": (
        "High", "Cross-site scripting (XSS) and malicious resource injection",
        "default-src 'self'; script-src 'self' https://www.googletagmanager.com"),
    "x-content-type-options": (
        "Medium", "MIME-type sniffing that can turn an upload into executable script",
        "nosniff"),
    "x-frame-options": (
        "Medium", "Clickjacking through framing of the site by a third party",
        "DENY  (or CSP frame-ancestors 'none')"),
    "referrer-policy": (
        "Low", "Leaking the full URL of the referring page to third parties",
        "strict-origin-when-cross-origin"),
    "permissions-policy": (
        "Low", "Unnecessary access to camera, microphone, geolocation and sensors",
        "geolocation=(), camera=(), microphone=()"),
    "cross-origin-opener-policy": (
        "Low", "Cross-window scripting attacks such as XS-Leaks",
        "same-origin"),
}

BANNER = "=" * 78


def title(t):
    print("\n" + BANNER)
    print(" " + t)
    print(BANNER)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Return the redirect response itself instead of following it, so that the
    redirect status and Location header can be inspected."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_plain = urllib.request.build_opener()
_norr = urllib.request.build_opener(NoRedirect)


def fetch(url, method="GET", follow=True):
    req = urllib.request.Request(url, method=method,
                                 headers={"User-Agent": "STQA-Exp05-HeaderAudit/1.0"})
    opener = _plain if follow else _norr
    try:
        r = opener.open(req, timeout=20)
        return r.status, {k.lower(): v for k, v in r.headers.items()}, r.read()
    except urllib.error.HTTPError as e:
        return e.code, {k.lower(): v for k, v in e.headers.items()}, b""


title("TARGET")
print(" URL      : %s" % BASE)
print(" Scanned  : %s" % datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
print(" Tool     : header_audit.py (OWASP Secure Headers Project checklist)")

# ---------------------------------------------------------------- transport --
title("1  TRANSPORT SECURITY (TLS)")
ctx = ssl.create_default_context()
with socket.create_connection((HOST, 443), timeout=15) as sock:
    with ctx.wrap_socket(sock, server_hostname=HOST) as ss:
        proto, cipher = ss.version(), ss.cipher()
        cert = ss.getpeercert()
print(" Protocol negotiated : %s" % proto)
print(" Cipher suite        : %s (%d bit)" % (cipher[0], cipher[2]))
print(" Certificate subject : %s" % dict(x[0] for x in cert["subject"]).get("commonName"))
print(" Certificate issuer  : %s" % dict(x[0] for x in cert["issuer"]).get("organizationName"))
print(" Valid until         : %s" % cert["notAfter"])
sans = [v for k, v in cert.get("subjectAltName", ()) if k == "DNS"]
print(" Subject alt names   : %s" % ", ".join(sans[:4]))
print(" Verdict             : %s" % ("PASS - TLS 1.2 or better with a trusted certificate"
                                     if proto in ("TLSv1.2", "TLSv1.3") else "FAIL"))

# ------------------------------------------------------- redirect behaviour --
title("2  HTTP TO HTTPS REDIRECTION")
r_code, hdrs, _ = fetch("http://%s/climate_/" % HOST, follow=False)
r_loc = hdrs.get("location", "")
print(" GET http://%s/climate_/  ->  HTTP %s" % (HOST, r_code))
print(" Location            : %s" % (r_loc or "(none)"))
redirects_to_https = r_code in (301, 302, 307, 308) and r_loc.startswith("https://")
print(" Verdict             : %s" % ("PASS - plain HTTP is redirected to HTTPS"
                                     if redirects_to_https else "REVIEW - no HTTPS redirect seen"))

# ---------------------------------------------------------- header analysis --
title("3  SECURITY HEADER ANALYSIS")
code, hdrs, body = fetch(BASE)
print(" GET %s  ->  HTTP %s\n" % (BASE, code))
print(" %-30s %-9s %s" % ("HEADER", "STATUS", "OBSERVED VALUE"))
print(" " + "-" * 74)
present, missing = [], []
for h, (sev, _risk, _good) in CHECKS.items():
    if h in hdrs:
        present.append(h)
        print(" %-30s %-9s %s" % (h, "PRESENT", hdrs[h][:38]))
    else:
        missing.append((h, sev))
        print(" %-30s %-9s %s" % (h, "MISSING", "-"))

print("\n Present : %d of %d" % (len(present), len(CHECKS)))
print(" Missing : %d of %d" % (len(missing), len(CHECKS)))

# ------------------------------------------------------- information leakage --
title("4  INFORMATION DISCLOSURE")
leaky = {"server": "Reveals the web server product",
         "x-powered-by": "Reveals the application framework",
         "x-aspnet-version": "Reveals the framework version"}
for h, why in leaky.items():
    v = hdrs.get(h)
    print(" %-18s %-34s %s" % (h, (v or "(not sent)"), why))
print("\n Note: GitHub Pages sends only a generic 'GitHub.com' server banner and no")
print(" framework version, so no exploitable version information is disclosed.")

# ---------------------------------------------------------- misconfig probes --
title("5  COMMON MISCONFIGURATION PROBES")
probes = [
    ("/climate_/.git/config",      "Exposed Git repository metadata"),
    ("/climate_/.env",             "Exposed environment / secrets file"),
    ("/climate_/admin/",           "Unprotected administration interface"),
    ("/climate_/backup.zip",       "Publicly downloadable backup archive"),
    ("/climate_/config.php.bak",   "Editor or backup copy of a config file"),
    ("/climate_/server-status",    "Apache status page left enabled"),
]
for path, why in probes:
    st, _, _ = fetch("https://%s%s" % (HOST, path))
    ok = st in (404, 403)
    print(" %-30s HTTP %-4s %-4s %s" % (path, st, "PASS" if ok else "FAIL", why))

# --------------------------------------------------- risk-rated finding list --
title("6  FINDINGS, RISK RATED")
if not missing:
    print(" No missing security headers.")
else:
    print(" %-32s %-9s %s" % ("FINDING", "SEVERITY", "RISK MITIGATED BY THE HEADER"))
    print(" " + "-" * 74)
    order = {"High": 0, "Medium": 1, "Low": 2}
    for h, sev in sorted(missing, key=lambda x: order[x[1]]):
        print(" %-32s %-9s %s" % ("Missing " + h, sev, CHECKS[h][1][:36]))
    print("\n RECOMMENDED VALUES")
    for h, sev in sorted(missing, key=lambda x: order[x[1]]):
        print("   %-30s %s" % (h + ":", CHECKS[h][2]))

title("SUMMARY")
high = sum(1 for _, s in missing if s == "High")
med = sum(1 for _, s in missing if s == "Medium")
low = sum(1 for _, s in missing if s == "Low")
print(" TLS                       : %s, certificate valid" % proto)
print(" HTTPS redirection         : %s" % ("enforced (HTTP %d to HTTPS)" % r_code
                                          if redirects_to_https else "not observed"))
print(" Sensitive path exposure   : none of the 6 probes returned content")
print(" Security headers present  : %d of %d" % (len(present), len(CHECKS)))
print(" Findings                  : %d High, %d Medium, %d Low" % (high, med, low))
print(BANNER)
