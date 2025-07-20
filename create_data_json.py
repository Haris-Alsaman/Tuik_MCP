import os
import json

def create_data_json():
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

    base_path = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_path, "data")
    
    all_data = []

    for kategori_name, kategori_path_name in kategoriler:
        kategori_dir = os.path.join(data_dir, kategori_path_name)
        
        if os.path.isdir(kategori_dir):
            files = [f for f in os.listdir(kategori_dir) if f.endswith('.xls') or f.endswith('.xlsx')]
            
            kategori_data = {
                "name": kategori_name,
                "kategori": kategori_path_name,
                "files": files
            }
            all_data.append(kategori_data)

    output_path = os.path.join(base_path, 'data.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)
    
    print(f"data.json has been created at {output_path}")

if __name__ == "__main__":
    create_data_json()