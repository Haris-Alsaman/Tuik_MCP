from tuik_scraper import TuikScraper
import os

def indir_kategori(kategori, kategori_path_name):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", kategori_path_name)
    os.makedirs(path, exist_ok=True)
    print(f"Downloading {kategori} to {path}")
    tuik = TuikScraper(download_folder_path=path)
    tuik.indir(kategori)

def indir_tum_kategoriler():
    kategoriler = [
        ('Adalet ve Seçim', 'adalet'),
        ('Bilim, Teknoloji ve Bilgi Toplumu', 'bilim'),
        ('Çevre ve Enerji', 'cevre'),
        ('Dış Ticaret', 'dis_ticaret'),
        ('Eğitim, Kültür, Spor ve Turizm', 'egitim'),
        ('Ekonomik Güven', 'ekonomik_guven'),
        ('Enflasyon ve Fiyat', 'enflasyon'),
        ('Gelir, Yaşam, Tüketim ve Yoksulluk', 'gelir'),
        ('İnşaat ve Konut', 'konut'),
        ('İstihdam, İşsizlik ve Ücret', 'istihdam'),
        ('Nüfus ve Demografi', 'nufus'),
        ('Sağlık ve Sosyal Koruma', 'saglik'),
        ('Sanayi', 'sanayi'),
        ('Tarım', 'tarim'),
        ('Ticaret ve Hizmet', 'ticaret'),
        ('Ulaştırma ve Haberleşme', 'ulastirma'),
        ('Ulusal Hesaplar', 'ulusal')
    ]
    for kategori, kategori_path_name in kategoriler:
        indir_kategori(kategori, kategori_path_name)

if __name__ == "__main__":
    indir_tum_kategoriler()
