"""Test konvensi format akhir (docs/SPEC.md bagian 5.2). Satu test per baris tabel + kasus tepi."""
import pytest

from normalize import normalize

REF_YEAR = 2026


def norm(text, **kw):
    return normalize(text, ref_year=REF_YEAR, **kw)


# --- Tabel SPEC 5.2 -------------------------------------------------------

@pytest.mark.parametrize("spoken, expected", [
    # Nominal uang
    ("lima ratus ribu rupiah", "Rp500.000"),
    # Nominal tanpa "rupiah"
    ("limitnya lima belas juta", "limitnya Rp15.000.000"),
    # Nominal campuran/pecahan
    ("dua setengah juta", "Rp2.500.000"),
    ("dua koma lima juta", "Rp2.500.000"),
    ("dua juta lima ratus ribu", "Rp2.500.000"),
    # Nominal desimal
    ("seribu lima ratus koma lima puluh", "Rp1.500,50"),
    # Persen/bunga
    ("nol koma lima persen", "0,5%"),
    # Tanggal lengkap
    ("tujuh belas Agustus sembilan belas delapan puluh lima", "17-08-1985"),
    # Tanggal, tahun 2 digit (dalam konteks tanggal lahir)
    ("tanggal lahir tujuh belas, nol delapan, delapan lima", "tanggal lahir 17-08-1985"),
    # Tanggal tanpa tahun
    ("tanggal dua lima", "tanggal 25"),
    # Masa berlaku kartu
    ("masa berlakunya nol delapan dua delapan", "masa berlakunya 08/28"),
    # Jam
    ("jam dua siang", "jam 14.00"),
    ("jam sepuluh lewat tiga puluh", "jam 10.30"),
    # Nomor HP
    ("nol delapan satu dua tiga empat lima enam tujuh delapan", "0812345678"),
    # "kosong" / "nol"
    ("kosong delapan satu dua tiga", "08123"),
    # "dobel" / "tripel"
    ("satu dobel tiga empat", "1334"),
    # Tenor/jumlah
    ("dua belas bulan", "12 bulan"),
    # Angka kecil non-entitas
    ("dua orang", "2 orang"),
    # Email
    ("andi titik budi at gmail titik com", "andi.budi@gmail.com"),
    # Nama & alamat
    ("Jalan Mawar nomor dua belas, RT tiga", "Jalan Mawar No. 12, RT 03"),
])
def test_spec_table(spoken, expected):
    assert norm(spoken).text == expected


def test_tanggal_dua_lima_bukan_deret_digit():
    # "dua lima" setelah "tanggal" = 25 (bukan 2 dan 5)
    assert norm("tanggal dua lima").text == "tanggal 25"


# --- Output ElevenLabs yang sudah berupa digit / format campuran ---------

@pytest.mark.parametrize("raw, expected", [
    ("Rp 500.000", "Rp500.000"),
    ("Rp. 500.000,-", "Rp500.000"),
    ("Rp500 ribu", "Rp500.000"),
    ("15 juta", "Rp15.000.000"),
    ("1,5 juta", "Rp1.500.000"),
    ("500.000 rupiah", "Rp500.000"),
    ("limitnya 15.000.000", "limitnya Rp15.000.000"),
    ("17 Agustus 1985", "17-08-1985"),
    ("17-08-1985", "17-08-1985"),
    ("17/8/85", "17-08-1985"),
    ("0,5 %", "0,5%"),
    ("jam 14:00", "jam 14.00"),
    ("0812-3456-7890", "081234567890"),
    ("0812 3456 7890", "081234567890"),
])
def test_elevenlabs_formats(raw, expected):
    assert norm(raw).text == expected


# --- Kasus tepi -----------------------------------------------------------

def test_nomor_dieja_berkelompok_dengan_koma():
    r = norm("nomor HP saya nol delapan satu dua, tiga empat lima enam, tujuh delapan sembilan nol")
    assert r.text == "nomor HP saya 081234567890"
    assert r.pii_types == ["no_hp"]


def test_koma_pendek_tidak_digabung():
    assert norm("pilih dua, tiga, atau empat").text == "pilih 2, 3, atau 4"


def test_tahun_bukan_uang():
    assert norm("sejak tahun dua ribu tiga").text == "sejak tahun 2003"


def test_ribu_poin_bukan_uang():
    assert norm("dapat sepuluh ribu poin").text == "dapat 10.000 poin"


def test_juta_diikuti_satuan_tidak_digabung():
    assert norm("cicilan dua juta tiga bulan").text == "cicilan Rp2.000.000 3 bulan"


def test_salah_satu_tetap_kata():
    assert norm("salah satu produk").text == "salah satu produk"


def test_angka_berdampingan_ambigu():
    r = norm("sekitar dua tiga hari")
    assert r.text == "sekitar dua tiga hari"
    assert "ambiguous_number" in r.needs_review


def test_jam_ambigu_ditandai():
    r = norm("saya telepon jam dua ya")
    assert r.text == "saya telepon jam 02.00 ya"
    assert "time_ambiguous" in r.needs_review


def test_jam_setengah_sore():
    assert norm("jam setengah tiga sore").text == "jam 14.30"


def test_koreksi_ditulis_semua():
    assert norm("delapan satu dua, eh, delapan satu tiga").text == "812, eh, 813"


def test_event_dibuang_dan_spasi_dirapikan():
    assert norm("iya , Pak (tertawa) baik").text == "iya, Pak baik"


def test_bulan_tanpa_tahun():
    assert norm("jatuh tempo tujuh belas agustus").text == "jatuh tempo 17 Agustus"


def test_tahun_dua_digit_abad():
    assert norm("lahir tanggal satu, nol satu, nol lima").text == "lahir tanggal 01-01-2005"


# --- Deteksi PII & secret --------------------------------------------------

def test_pii_nominal_dan_tanggal():
    r = norm("limitnya lima belas juta, tanggal lahir Ibu tujuh belas Agustus sembilan belas delapan puluh lima")
    assert r.text == "limitnya Rp15.000.000, tanggal lahir Ibu 17-08-1985"
    assert r.pii_types == ["nominal", "tanggal"]


def test_pii_nik_dan_kartu():
    assert norm("NIK saya 3171234567890001").pii_types == ["nik"]
    assert norm("nomor kartunya 4111111111111111").pii_types == ["no_kartu"]


def test_pii_nomor_di_akhir_kalimat():
    assert norm("Nomor HP yang terdaftar? 0812 3456 7890.").pii_types == ["no_hp"]


def test_pii_nama_dari_crm():
    assert "nama" in norm("dengan Ibu Sari Wulandari?", crm_name="Sari Wulandari").pii_types


def test_secret_hanya_bila_nilai_disebut():
    assert norm("kode OTP nya satu dua tiga empat lima enam").secret
    assert norm("PIN saya 1234").secret
    # instruksi / pertanyaan tanpa nilai: bukan rahasia, tapi tercatat sebagai mention
    r = norm("boleh sebutkan kode OTP nya Bu")
    assert not r.secret and r.secret_mention
    assert not norm("silakan ketik PIN spasi enam angka").secret
    assert norm("tiga angka di belakang kartu").secret_mention
    # nominal / tanggal tidak dianggap nilai rahasia
    assert not norm("PIN aktif, limit lima belas juta, lahir 17 Agustus 1985").secret
    assert not norm("limitnya lima belas juta").secret_mention
