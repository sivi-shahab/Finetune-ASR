"""Normalisasi teks transkrip ElevenLabs ke format akhir label training.

Aturan mengikuti docs/SPEC.md bagian 5. Alurnya:

1. Bersihkan event non-speech dan rapikan email yang dieja.
2. Tokenisasi, lalu ubah kata bilangan / token angka menjadi token ``num``.
3. Terapkan aturan kontekstual berurutan: jam, tanggal, masa berlaku kartu,
   persen, nominal uang, RT/RW/nomor alamat, penggabungan nomor identitas.
4. Render sisa angka sebagai digit, susun ulang teks, deteksi ``pii_types``,
   ``secret``, dan alasan ``needs_review``.

Pemakaian::

    from normalize import normalize
    r = normalize("limitnya lima belas juta ya Bu")
    r.text        # "limitnya Rp15.000.000 ya Bu"
    r.pii_types   # ["nominal"]
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from decimal import ROUND_HALF_UP, Decimal

# ---------------------------------------------------------------------------
# Kosakata
# ---------------------------------------------------------------------------

UNITS = {
    "nol": 0, "satu": 1, "dua": 2, "tiga": 3, "empat": 4, "lima": 5,
    "enam": 6, "tujuh": 7, "delapan": 8, "lapan": 8, "sembilan": 9,
}
DIGIT_WORDS = {**{k: str(v) for k, v in UNITS.items()}, "kosong": "0"}
REPEAT_WORDS = {"dobel": 2, "double": 2, "tripel": 3, "triple": 3}
COMPOUND_WORDS = {"belas", "puluh", "ratus"}
SCALES = {"ribu": 10**3, "juta": 10**6, "miliar": 10**9, "milyar": 10**9, "triliun": 10**12}
SE_SCALES = {"seribu": 10**3, "sejuta": 10**6, "semiliar": 10**9, "semilyar": 10**9, "setriliun": 10**12}

MONTHS = {
    "januari": 1, "februari": 2, "pebruari": 2, "maret": 3, "april": 4, "mei": 5,
    "juni": 6, "juli": 7, "agustus": 8, "september": 9, "oktober": 10,
    "november": 11, "nopember": 11, "desember": 12,
}
MONTH_NAMES = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli",
               "Agustus", "September", "Oktober", "November", "Desember"]

TIME_WORDS = {"jam", "pukul"}
TIME_QUALIFIERS = {"pagi", "siang", "sore", "malam"}
DATE_CONTEXT = {"lahir", "tanggal", "tgl"}
EXPIRY_CONTEXT = {"berlaku", "expired", "expire", "exp", "kedaluwarsa", "kadaluarsa", "kadaluwarsa", "valid"}
ADDRESS_CONTEXT = {"jalan", "jl", "jln", "gang", "gg", "blok", "perumahan", "perum", "komplek", "kompleks"}
MONEY_CONTEXT = {
    "limit", "limitnya", "tagihan", "tagihannya", "cicilan", "cicilannya", "biaya", "biayanya",
    "iuran", "plafon", "plafonnya", "pinjaman", "pinjamannya", "saldo", "saldonya", "bayar",
    "membayar", "dibayar", "nominal", "nominalnya", "harga", "harganya", "gaji", "penghasilan",
    "cashback", "dana", "transfer", "denda", "fee", "angsuran", "angsurannya",
}
# Satuan yang menandakan angka berskala (ribu/juta) BUKAN uang.
NON_MONEY_UNITS = {
    "orang", "kali", "poin", "point", "km", "kilometer", "meter", "nasabah", "jiwa", "unit",
    "lembar", "transaksi", "tahun", "bulan", "hari", "jam", "menit", "detik", "kartu",
    "peserta", "pengguna", "member", "persen",
}
CONTEXT_WINDOW = 6

# ---------------------------------------------------------------------------
# Struktur token
# ---------------------------------------------------------------------------


@dataclass
class Num:
    value: Decimal
    frac: str = ""                 # digit desimal sesuai ucapan (untuk format)
    has_scale: bool = False        # memakai ribu/juta/miliar/triliun
    digits: str | None = None      # deret digit yang harus dipertahankan apa adanya
    single_unit: bool = False      # satu kata satuan 0-9, mis. "dua"
    src: str = ""                  # teks asli

    @property
    def is_int(self) -> bool:
        return self.value == self.value.to_integral_value() and not self.frac


@dataclass
class Tok:
    text: str
    kind: str                      # word | punct | num | time | date | final
    num: Num | None = None
    extra: tuple = ()

    @property
    def low(self) -> str:
        return self.text.lower() if self.kind == "word" else ""


@dataclass
class NormResult:
    text: str
    pii_types: list[str] = field(default_factory=list)
    secret: bool = False           # nilai rahasia (PIN/OTP/CVV/password) kemungkinan disebut di segmen ini
    secret_mention: bool = False   # kata kunci rahasia disebut (mis. instruksi "buat PIN"), belum tentu nilainya
    needs_review: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pra-proses
# ---------------------------------------------------------------------------

EVENT_RE = re.compile(r"[\(\[][^\)\]]*[\)\]]")
EMAIL_SPOKEN_RE = re.compile(
    r"\b([a-z0-9]+(?:\s+(?:titik|dot|underscore|strip)\s+[a-z0-9]+)*)"
    r"\s+(?:at|et|@)\s+"
    r"([a-z0-9]+(?:\s+(?:titik|dot)\s+[a-z0-9]+)+)\b",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r"Rp(?=\s?\d)|[^\W_]+(?:[.,:/@_\-][^\W_]+)*|[^\w\s]|_")


def _spoken_email(m: re.Match) -> str:
    def join(s: str) -> str:
        s = re.sub(r"\s+(?:titik|dot)\s+", ".", s, flags=re.IGNORECASE)
        s = re.sub(r"\s+underscore\s+", "_", s, flags=re.IGNORECASE)
        s = re.sub(r"\s+strip\s+", "-", s, flags=re.IGNORECASE)
        return s.lower()
    return f"{join(m.group(1))}@{join(m.group(2))}"


def _tokenize(text: str) -> list[Tok]:
    toks = []
    for m in TOKEN_RE.finditer(text):
        s = m.group(0)
        kind = "word" if (s[0].isalnum() or s == "Rp") else "punct"
        toks.append(Tok(s, kind))
    return toks


# ---------------------------------------------------------------------------
# Parsing bilangan
# ---------------------------------------------------------------------------

THOUSANDS_RE = re.compile(r"^\d{1,3}(?:\.\d{3})+$")
THOUSANDS_DEC_RE = re.compile(r"^(\d{1,3}(?:\.\d{3})*),(\d+)$")
EN_THOUSANDS_RE = re.compile(r"^\d{1,3}(?:,\d{3}){2,}$")
EN_DEC_RE = re.compile(r"^(\d+)\.(\d{1,2})$")
TIME_TOKEN_RE = re.compile(r"^(\d{1,2})[.:](\d{2})$")
DATE_TOKEN_RE = re.compile(r"^(\d{1,2})[-/.](\d{1,2})[-/.](\d{4}|\d{2})$")
ID_TOKEN_RE = re.compile(r"^\d+(?:[-.]\d+)+$")
EXPIRY_TOKEN_RE = re.compile(r"^(\d{2})[/-](\d{2})$")


def parse_digit_token(s: str) -> Num | None:
    """Token yang sudah berupa digit (output ElevenLabs), mis. ``15``, ``1,5``, ``500.000``."""
    if s.isdigit():
        if (len(s) > 1 and s[0] == "0") or len(s) >= 5:
            return Num(Decimal(s), digits=s, src=s)
        return Num(Decimal(s), src=s)
    if THOUSANDS_RE.match(s):
        return Num(Decimal(s.replace(".", "")), src=s)
    if EN_THOUSANDS_RE.match(s):
        return Num(Decimal(s.replace(",", "")), src=s)
    m = THOUSANDS_DEC_RE.match(s) or EN_DEC_RE.match(s)
    if m:
        whole, frac = m.group(1).replace(".", ""), m.group(2)
        return Num(Decimal(f"{whole}.{frac}"), frac=frac, src=s)
    return None


class _Parser:
    """Mengubah kata bilangan di daftar token menjadi token ``num``."""

    def __init__(self, toks: list[Tok]):
        self.t = toks

    def w(self, j: int) -> str:
        return self.t[j].low if 0 <= j < len(self.t) else ""

    def group(self, j: int) -> tuple[int, int] | None:
        """Kelompok 0..999 dari kata, mis. 'sembilan ratus delapan puluh lima'."""
        if self.w(j) == "nol":
            return 0, j + 1
        v, start = 0, j
        if self.w(j) == "seratus":
            v, j = 100, j + 1
        elif UNITS.get(self.w(j), 0) > 0 and self.w(j + 1) == "ratus":
            v, j = UNITS[self.w(j)] * 100, j + 2
        u = UNITS.get(self.w(j), 0)
        if self.w(j) == "sepuluh":
            v, j = v + 10, j + 1
        elif self.w(j) == "sebelas":
            v, j = v + 11, j + 1
        elif u > 0 and self.w(j + 1) == "belas":
            v, j = v + 10 + u, j + 2
        elif u > 0 and self.w(j + 1) == "puluh":
            v, j = v + 10 * u, j + 2
            u2 = UNITS.get(self.w(j), 0)
            if u2 > 0 and self.w(j + 1) not in COMPOUND_WORDS:
                v, j = v + u2, j + 1
        elif u > 0 and self.w(j + 1) not in COMPOUND_WORDS:
            v, j = v + u, j + 1
        return (v, j) if j > start else None

    def digit_run(self, i: int) -> tuple[Num, int] | None:
        """Deret digit yang dieja satu per satu, mis. 'nol delapan satu dua'."""
        digits, j, cut = "", i, None
        while True:
            lw = self.w(j)
            if lw in REPEAT_WORDS and self.w(j + 1) in DIGIT_WORDS:
                digits += DIGIT_WORDS[self.w(j + 1)] * REPEAT_WORDS[lw]
                j += 2
            elif lw in DIGIT_WORDS and self.w(j + 1) not in COMPOUND_WORDS:
                digits += DIGIT_WORDS[lw]
                j += 1
            elif (digits and j < len(self.t) and self.t[j].text == ","
                  and (self.w(j + 1) in REPEAT_WORDS
                       or (self.w(j + 1) in DIGIT_WORDS and self.w(j + 2) not in COMPOUND_WORDS))):
                if cut is None:
                    cut = (digits, j)
                j += 1
            else:
                break
        # Melewati koma hanya sah untuk nomor panjang; "dua, tiga" tetap dua angka.
        if cut is not None and len(digits) < 6:
            digits, j = cut
        if len(digits) >= 3 or (len(digits) == 2 and digits[0] == "0"):
            src = " ".join(x.text for x in self.t[i:j])
            return Num(Decimal(digits), digits=digits, src=src), j
        return None

    def cardinal(self, i: int) -> tuple[Num, int] | None:
        total, j = Decimal(0), i
        last_scale: int | None = None
        has_scale, frac, got, single_unit = False, "", False, False
        while True:
            start = j
            lw = self.w(j)
            if lw in SE_SCALES:
                s = SE_SCALES[lw]
                if last_scale is not None and s >= last_scale:
                    break
                total, last_scale, has_scale, got, j = total + s, s, True, True, j + 1
                continue
            lf = ""
            if lw == "setengah" and self.w(j + 1) in SCALES:
                g = Decimal("0.5")
                j += 1
            else:
                r = self.group(j)
                if r:
                    g, j = Decimal(r[0]), r[1]
                elif j < len(self.t) and self.t[j].kind == "word" and self.t[j].text[0].isdigit():
                    n = parse_digit_token(self.t[j].text)
                    if n is None:
                        break
                    if n.digits is None and n.value < 1000:
                        g, lf, j = n.value, n.frac, j + 1
                    elif not got:
                        return n, j + 1
                    else:
                        break
                else:
                    break
            if self.w(j) == "setengah" and g == g.to_integral_value():
                g, j = g + Decimal("0.5"), j + 1
            elif self.w(j) == "koma":
                d, k = "", j + 1
                while self.w(k) in UNITS and self.w(k + 1) not in COMPOUND_WORDS:
                    d, k = d + str(UNITS[self.w(k)]), k + 1
                if not d:
                    r = self.group(j + 1)
                    if r:
                        d, k = str(r[0]), r[1]
                if d:
                    g, lf, j = Decimal(f"{int(g)}.{d}"), d, k
            lw = self.w(j)
            if lw in SCALES and (last_scale is None or SCALES[lw] < last_scale):
                total += g * SCALES[lw]
                last_scale, has_scale, got = SCALES[lw], True, True
                j += 1
                continue
            if got:
                # Sisa tanpa skala setelah ribu boleh apa saja ("dua ribu tiga" = 2003);
                # setelah juta ke atas harus ratusan ("dua juta lima ratus").
                if last_scale == 1000 or g >= 100:
                    total += g
                    frac = lf
                else:
                    j = start
                break
            total, frac, got = g, lf, True
            single_unit = (j == start + 1 and self.t[start].low in UNITS)
            break
        if not got:
            return None
        src = " ".join(x.text for x in self.t[i:j])
        return Num(total, frac=frac, has_scale=has_scale, single_unit=single_unit, src=src), j

    def run(self) -> list[Tok]:
        out, i = [], 0
        while i < len(self.t):
            tok = self.t[i]
            if tok.kind != "word":
                out.append(tok)
                i += 1
                continue
            prev = out[-1].low if out else ""
            m = TIME_TOKEN_RE.match(tok.text)
            if m and prev in TIME_WORDS:
                out.append(Tok(tok.text, "time", extra=(int(m.group(1)), int(m.group(2)))))
                i += 1
                continue
            m = DATE_TOKEN_RE.match(tok.text)
            if m:
                out.append(Tok(tok.text, "date", extra=tuple(m.groups())))
                i += 1
                continue
            if ID_TOKEN_RE.match(tok.text) and not THOUSANDS_RE.match(tok.text):
                d = re.sub(r"\D", "", tok.text)
                if len(d) >= 8:
                    out.append(Tok(tok.text, "num", Num(Decimal(d), digits=d, src=tok.text)))
                    i += 1
                    continue
            r = self.digit_run(i) or self.cardinal(i)
            if r:
                out.append(Tok(r[0].src, "num", r[0]))
                i = r[1]
            else:
                out.append(tok)
                i += 1
        return out


# ---------------------------------------------------------------------------
# Format
# ---------------------------------------------------------------------------


def _dots(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def fmt_plain(n: Num) -> str:
    if n.digits is not None:
        return n.digits
    if n.frac:
        return f"{int(n.value)},{n.frac}"
    if n.value != n.value.to_integral_value():
        return f"{n.value.normalize():f}".replace(".", ",")
    v = int(n.value)
    # Titik ribuan hanya untuk angka besar berskala; tahun "dua ribu tiga" tetap 2003.
    return _dots(v) if n.has_scale and v >= 10000 else str(v)


def fmt_money(n: Num) -> str:
    if n.frac:
        return f"Rp{_dots(int(n.value))},{n.frac}"
    v = int(n.value.quantize(Decimal(1), rounding=ROUND_HALF_UP))
    return f"Rp{_dots(v)}"


def expand_year(yy: int, ref_year: int) -> int:
    return 1900 + yy if yy > ref_year % 100 else 2000 + yy


# ---------------------------------------------------------------------------
# Aturan kontekstual
# ---------------------------------------------------------------------------


class _Rules:
    def __init__(self, toks: list[Tok], ref_year: int):
        self.t = toks
        self.ref_year = ref_year
        self.review: list[str] = []

    # -- util --
    def isnum(self, j: int, int_only: bool = True) -> bool:
        if not (0 <= j < len(self.t)) or self.t[j].kind != "num":
            return False
        n = self.t[j].num
        return (n.is_int and not n.has_scale) if int_only else True

    def val(self, j: int) -> int:
        return int(self.t[j].num.value)

    def w(self, j: int) -> str:
        return self.t[j].low if 0 <= j < len(self.t) else ""

    def context(self, i: int, words: set[str]) -> bool:
        for k in range(max(0, i - CONTEXT_WINDOW), i):
            lw = self.w(k).strip(".")
            if lw in words or lw.removesuffix("nya") in words:
                return True
        return False

    def skip_sep(self, j: int, seps: str = ",-/.") -> int:
        if 0 <= j < len(self.t) and self.t[j].kind == "punct" and self.t[j].text in seps:
            return j + 1
        return j

    def replace(self, i: int, j: int, text: str) -> None:
        self.t[i:j] = [Tok(text, "final")]

    def year_at(self, j: int) -> tuple[int, int] | None:
        """Tahun mulai di token j: 1985 | 19 85 | 8 5 (dua digit dieja) | 85."""
        if not self.isnum(j):
            return None
        n, v = self.t[j].num, self.val(j)
        if n.digits is not None and len(n.digits) == 4:
            return int(n.digits), j + 1
        if 1000 <= v <= 2100:
            return v, j + 1
        if v in (19, 20) and self.isnum(j + 1):
            if self.t[j + 1].num.single_unit and self.isnum(j + 2) and self.t[j + 2].num.single_unit:
                return v * 100 + self.val(j + 1) * 10 + self.val(j + 2), j + 3
            if self.val(j + 1) <= 99:
                return v * 100 + self.val(j + 1), j + 2
        if n.single_unit and self.isnum(j + 1) and self.t[j + 1].num.single_unit:
            return expand_year(v * 10 + self.val(j + 1), self.ref_year), j + 2
        if v <= 99:
            return expand_year(v, self.ref_year), j + 1
        return None

    # -- aturan --
    def times(self) -> None:
        i = 0
        while i < len(self.t):
            if self.w(i) not in TIME_WORDS:
                i += 1
                continue
            j, h, m, raw24 = i + 1, None, 0, False
            if j < len(self.t) and self.t[j].kind == "time":
                h, m = self.t[j].extra
                raw24 = h >= 13
                j += 1
            elif self.w(j) == "setengah" and self.isnum(j + 1) and 1 <= self.val(j + 1) <= 12:
                h, m, j = (self.val(j + 1) - 1) or 12, 30, j + 2
            elif self.isnum(j) and 0 <= self.val(j) <= 24:
                h, j = self.val(j), j + 1
                if self.w(j) in ("lewat", "lebih") and self.isnum(j + 1) and self.val(j + 1) <= 59:
                    m, j = self.val(j + 1), j + 2
                    if self.w(j) == "menit":
                        j += 1
                elif self.isnum(j) and 10 <= self.val(j) <= 59:
                    m, j = self.val(j), j + 1
            if h is None:
                i += 1
                continue
            q = self.w(j)
            if q in TIME_QUALIFIERS:
                if q == "siang" and 1 <= h <= 5:
                    h += 12
                elif q == "sore" and 1 <= h <= 11:
                    h += 12
                elif q == "malam" and 6 <= h <= 11:
                    h += 12
                elif q == "malam" and h == 12:
                    h = 0
                j += 1
            elif 1 <= h <= 6 and not raw24:
                self.review.append("time_ambiguous")
            self.replace(i + 1, j, f"{h:02d}.{m:02d}")
            i += 2

    def dates(self) -> None:
        # "tanggal dua lima" = tanggal 25: dua digit dieja setelah "tanggal" adalah satu angka.
        for i in range(1, len(self.t) - 1):
            if (self.w(i - 1) in ("tanggal", "tgl") and self.isnum(i) and self.isnum(i + 1)
                    and self.t[i].num.single_unit and self.t[i + 1].num.single_unit
                    and 1 <= self.val(i) * 10 + self.val(i + 1) <= 31):
                v = self.val(i) * 10 + self.val(i + 1)
                src = f"{self.t[i].num.src} {self.t[i + 1].num.src}"
                self.t[i:i + 2] = [Tok(src, "num", Num(Decimal(v), src=src))]
        i = 0
        while i < len(self.t):
            tok = self.t[i]
            if tok.kind == "date":
                d, mo, y = (int(x) for x in tok.extra)
                y = expand_year(y, self.ref_year) if len(tok.extra[2]) == 2 else y
                if 1 <= d <= 31 and 1 <= mo <= 12:
                    self.replace(i, i + 1, f"{d:02d}-{mo:02d}-{y}")
                else:
                    self.review.append("date_invalid")
                i += 1
                continue
            if self.isnum(i) and 1 <= self.val(i) <= 31:
                # (a) 17 Agustus 1985
                j = self.skip_sep(i + 1, ",")
                if self.w(j) in MONTHS:
                    mo = MONTHS[self.w(j)]
                    k = self.skip_sep(j + 1, ",")
                    if self.w(k) == "tahun":
                        k += 1
                    y = self.year_at(k)
                    if y:
                        self.replace(i, y[1], f"{self.val(i):02d}-{mo:02d}-{y[0]}")
                    else:
                        self.replace(i, j + 1, f"{self.val(i)} {MONTH_NAMES[mo - 1]}")
                    i += 1
                    continue
                # (b) 17, 08, 85 dalam konteks tanggal lahir
                if self.context(i, DATE_CONTEXT):
                    j = self.skip_sep(i + 1)
                    if self.isnum(j) and 1 <= self.val(j) <= 12:
                        k = self.skip_sep(j + 1)
                        y = self.year_at(k)
                        if y and y[0] >= 1900:
                            self.replace(i, y[1], f"{self.val(i):02d}-{self.val(j):02d}-{y[0]}")
                            i += 1
                            continue
            i += 1

    def expiry(self) -> None:
        for i in range(len(self.t)):
            if i >= len(self.t) or not self.context(i, EXPIRY_CONTEXT):
                continue
            tok = self.t[i]
            m = EXPIRY_TOKEN_RE.match(tok.text) if tok.kind == "word" else None
            if m and 1 <= int(m.group(1)) <= 12:
                self.replace(i, i + 1, f"{m.group(1)}/{m.group(2)}")
            elif tok.kind == "num" and tok.num.digits and len(tok.num.digits) == 4:
                d = tok.num.digits
                if 1 <= int(d[:2]) <= 12:
                    self.replace(i, i + 1, f"{d[:2]}/{d[2:]}")
            elif self.isnum(i) and 1 <= self.val(i) <= 12:
                k = self.skip_sep(i + 1)
                y = self.year_at(k)
                if y:
                    self.replace(i, y[1], f"{self.val(i):02d}/{y[0] % 100:02d}")

    def percent(self) -> None:
        i = 0
        while i < len(self.t):
            if self.isnum(i, int_only=False) and (self.w(i + 1) == "persen"
                                                  or (i + 1 < len(self.t) and self.t[i + 1].text == "%")):
                self.replace(i, i + 2, fmt_plain(self.t[i].num) + "%")
            i += 1

    def money(self) -> None:
        i = 0
        while i < len(self.t):
            if not self.isnum(i, int_only=False) or self.t[i].num.digits and not self._money_hint(i):
                i += 1
                continue
            n = self.t[i].num
            start, end = i, i + 1
            is_money = False
            # "Rp" atau "Rp." sebelum angka
            p = i - 1
            if p >= 0 and self.t[p].text == ".":
                p -= 1
            if p >= 0 and self.w(p) == "rp":
                start, is_money = p, True
            nxt = self.w(i + 1)
            if nxt in ("rupiah", "perak"):
                end, is_money = i + 2, True
            elif n.has_scale and nxt not in NON_MONEY_UNITS and self.w(i - 1) not in ("tahun", "thn"):
                is_money = True
            elif self.context(i, MONEY_CONTEXT) and n.value >= 1000 and n.digits is None:
                is_money = True
            if not is_money:
                i += 1
                continue
            # buang ",-" di belakang nominal (gaya "Rp500.000,-")
            if (end + 1 < len(self.t) and self.t[end].text == "," and self.t[end + 1].text == "-"):
                end += 2
            self.replace(start, end, fmt_money(n))
            i = start + 1

    def _money_hint(self, i: int) -> bool:
        p = i - 1
        if p >= 0 and self.t[p].text == ".":
            p -= 1
        return self.w(p) == "rp" or self.w(i + 1) in ("rupiah", "perak")

    def address(self) -> None:
        i = 0
        while i < len(self.t):
            lw = self.w(i)
            if lw in ("rt", "rw") and self.isnum(i + 1) and self.val(i + 1) < 1000:
                self.t[i] = Tok(lw.upper(), "final")
                self.replace(i + 1, i + 2, f"{self.val(i + 1):02d}")
            elif lw in ("nomor", "no") and self.context(i, ADDRESS_CONTEXT):
                j = i + 1
                if j < len(self.t) and self.t[j].text == ".":
                    j += 1
                if self.isnum(j) and self.val(j) < 10000:
                    self.t[i:j] = [Tok("No.", "final")]
            i += 1

    def join_ids(self) -> None:
        def piece(j: int) -> str | None:
            if not (0 <= j < len(self.t)) or self.t[j].kind != "num":
                return None
            n = self.t[j].num
            if n.digits is not None:
                return n.digits
            if n.src.isdigit() and len(n.src) >= 3:
                return n.src
            return None

        i = 0
        while i < len(self.t):
            first = piece(i)
            if first is None:
                i += 1
                continue
            parts, j, end = [first], i + 1, i + 1
            while True:
                k = self.skip_sep(j, ",-")
                p = piece(k)
                if p is None:
                    break
                parts.append(p)
                j = end = k + 1
            joined = "".join(parts)
            if len(parts) >= 2 and len(joined) >= 8:
                self.t[i:end] = [Tok(joined, "num", Num(Decimal(joined), digits=joined, src=joined))]
            i += 1

    def render(self) -> list[Tok]:
        out = []
        for i, tok in enumerate(self.t):
            if tok.kind != "num":
                out.append(tok)
                continue
            n = tok.num
            if n.single_unit and n.src.lower() == "satu" and (
                    self.w(i - 1) == "salah" or self.w(i + 1) in ("per", "sama")):
                out.append(Tok(n.src, "word"))
                continue
            adjacent = (i + 1 < len(self.t) and self.t[i + 1].kind == "num" and self.t[i + 1].num.single_unit)
            adjacent_prev = (i > 0 and self.t[i - 1].kind == "num" and self.t[i - 1].num.single_unit)
            if n.single_unit and (adjacent or adjacent_prev):
                # "dua tiga hari" (2–3 hari) ambigu; biarkan reviewer memutuskan.
                out.append(Tok(n.src, "word"))
                self.review.append("ambiguous_number")
                continue
            out.append(Tok(fmt_plain(n), "final"))
        return out


# ---------------------------------------------------------------------------
# Susun ulang & deteksi
# ---------------------------------------------------------------------------

NO_SPACE_BEFORE = set(".,!?;:)]}%…")
NO_SPACE_AFTER = set("([{")


def _join(toks: list[Tok]) -> str:
    out = ""
    for t in toks:
        s = t.text
        if not out:
            out = s
        elif (t.kind == "punct" and s in NO_SPACE_BEFORE) or out[-1] in NO_SPACE_AFTER:
            out += s
        else:
            out += " " + s
    return out


SECRET_RE = re.compile(
    r"\b(otp|pin|cvv|cvc|password|kata sandi|kode verifikasi|kode otp|"
    r"(tiga|3) (angka|digit) (di )?(belakang|balik))\b", re.IGNORECASE)
ADDRESS_RE = re.compile(
    r"\b(jalan|jl\.?|jln\.?|gang|gg\.|blok|RT \d{2}|RW \d{2}|kelurahan|kecamatan|kel\.|kec\.|"
    r"perumahan|komplek|kompleks)\b", re.IGNORECASE)
LONG_DIGITS_RE = re.compile(r"(?<![\d.,])(\+?\d{8,})(?!\d|[.,]\d)")


def luhn_ok(d: str) -> bool:
    total, alt = 0, False
    for ch in reversed(d):
        x = int(ch)
        if alt:
            x = x * 2 - 9 if x > 4 else x * 2
        total += x
        alt = not alt
    return total % 10 == 0


ENTITY_PATTERNS = (
    ("nominal", re.compile(r"Rp\d[\d.]*(?:,\d+)?")),
    ("tanggal", re.compile(r"\b\d{2}-\d{2}-\d{4}\b|\b\d{2}/\d{2}\b")),
    ("email", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
)


def _id_type(digits: str, before: str) -> tuple[str, bool]:
    """Jenis nomor identitas dari kata kunci sebelumnya; (jenis, perlu_review)."""
    if re.search(r"\b(nik|ktp)\b", before):
        return "nik", False
    if re.search(r"\b(rekening|rek)\b", before):
        return "no_rekening", False
    if re.search(r"\b(kartu|card)\b", before):
        return "no_kartu", False
    if re.search(r"\b(hp|handphone|telepon|telp|whatsapp|wa|ponsel)\b", before) or \
            digits.startswith(("08", "628", "+628")):
        return "no_hp", False
    if len(digits) in (15, 16) and luhn_ok(digits):
        return "no_kartu", False
    return ("nik" if len(digits) == 16 else "no_rekening"), True


def extract_entities(text: str) -> list[tuple[str, str]]:
    """Entitas berformat akhir di teks: [(jenis, nilai)], urut kemunculan.
    Dipakai untuk pii_types dan metrik evaluasi (src/evaluate.py)."""
    found = []
    for kind, pat in ENTITY_PATTERNS:
        found += [(m.start(), kind, m.group(0).rstrip(".,")) for m in pat.finditer(text)]
    low = text.lower()
    for m in LONG_DIGITS_RE.finditer(text):
        kind, _ = _id_type(m.group(1), low[max(0, m.start() - 60):m.start()])
        found.append((m.start(), kind, m.group(1)))
    return [(k, v) for _, k, v in sorted(found)]


def detect_pii(text: str, crm_name: str | None = None) -> tuple[list[str], list[str]]:
    """Kembalikan (pii_types, alasan review) dari teks yang sudah dinormalisasi."""
    types = {k for k, _ in extract_entities(text)}
    low = text.lower()
    if ADDRESS_RE.search(text):
        types.add("alamat")
    if (crm_name and crm_name.lower() in low) or "atas nama" in low:
        types.add("nama")
    review = []
    for m in LONG_DIGITS_RE.finditer(text):
        if _id_type(m.group(1), low[max(0, m.start() - 60):m.start()])[1]:
            review.append("id_unknown")
    return sorted(types), review


SECRET_VALUE_RE = re.compile(r"(?<![\d.,])\d{3,}(?![\d.,]?\d)")


def has_secret_value(text: str) -> bool:
    """Ada deret >= 3 digit yang bukan nominal (Rp...), tanggal, atau jam."""
    t = re.sub(r"Rp\d[\d.]*(?:,\d+)?|\b\d{2}-\d{2}-\d{4}\b|\b\d{2}/\d{2}\b|\b\d{2}\.\d{2}\b", " ", text)
    return bool(SECRET_VALUE_RE.search(t))


def normalize(text: str, *, crm_name: str | None = None, ref_year: int | None = None) -> NormResult:
    """Ubah teks transkrip ke format akhir label training (docs/SPEC.md bagian 5)."""
    ref_year = ref_year or date.today().year
    mention = bool(SECRET_RE.search(text))
    text = EVENT_RE.sub(" ", text)
    text = EMAIL_SPOKEN_RE.sub(_spoken_email, text)
    toks = _Parser(_tokenize(text)).run()
    rules = _Rules(toks, ref_year)
    rules.times()
    rules.dates()
    rules.expiry()
    rules.percent()
    rules.money()
    rules.address()
    rules.join_ids()
    out = _join(rules.render())
    out = re.sub(r"\s+", " ", out).strip()
    pii, review = detect_pii(out, crm_name)
    return NormResult(
        text=out,
        pii_types=pii,
        # Rahasia hanya bila kata kunci DAN nilainya muncul; "buat PIN" (instruksi) tidak dibuang.
        secret=(mention or bool(SECRET_RE.search(out))) and has_secret_value(out),
        secret_mention=mention or bool(SECRET_RE.search(out)),
        needs_review=sorted(set(rules.review + review)),
    )
