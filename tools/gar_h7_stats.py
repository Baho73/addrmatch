"""H7: сколько уникальных имён улиц в ГАР. Читает AS_ADDR_OBJ_* всех регионов прямо из zip.

Требует архив ГАР gar_xml.zip (~57 ГБ, fias.nalog.ru); результат прогона сохранён в
docs/gar-h7-result.json — повторный запуск без архива не нужен, если числа не изменились."""
import zipfile, re, collections, json, sys, time
import xml.etree.ElementTree as ET

ZIP = r"F:/Downloads/gar_xml.zip"  # архив ГАР (ФНС, fias.nalog.ru)
OUT = "docs/gar-h7-result.json"

def norm(s):
    return re.sub(r"\s+", " ", s.lower().replace("ё", "е")).strip()

t0 = time.time()
z = zipfile.ZipFile(ZIP)
files = [i for i in z.infolist() if re.search(r"^\d+/AS_ADDR_OBJ_\d{8}", i.filename)]
levels = collections.Counter()          # LEVEL -> objects (актуальные)
street_names = collections.Counter()    # нормализованное имя улицы -> число объектов
street_types = collections.Counter()    # TYPENAME на уровне улиц
city_names = collections.Counter()      # имена населённых пунктов (уровни 5,6)
per_region_streets = {}
for k, fi in enumerate(files, 1):
    reg = fi.filename.split("/")[0]
    n_streets = 0
    with z.open(fi) as fh:
        for _, el in ET.iterparse(fh, events=("end",)):
            if el.tag != "OBJECT":
                continue
            a = el.attrib
            if a.get("ISACTUAL") == "1" and a.get("ISACTIVE") == "1":
                lvl = a.get("LEVEL")
                levels[lvl] += 1
                name = norm(a.get("NAME", ""))
                if lvl == "8":
                    street_names[name] += 1
                    street_types[a.get("TYPENAME", "")] += 1
                    n_streets += 1
                elif lvl in ("5", "6"):
                    city_names[name] += 1
            el.clear()
    per_region_streets[reg] = n_streets
    print(f"[{k}/{len(files)}] region {reg}: streets={n_streets} uniq_so_far={len(street_names)} t={time.time()-t0:.0f}s", flush=True)

res = {
    "regions": len(files),
    "levels_actual": dict(levels),
    "streets_total": sum(street_names.values()),
    "streets_unique_names": len(street_names),
    "street_types_unique": len(street_types),
    "street_types_top": street_types.most_common(30),
    "top_street_names": street_names.most_common(40),
    "cities_total_lvl56": sum(city_names.values()),
    "cities_unique_names": len(city_names),
    "names_covering_50pct": None,
    "per_region_streets": per_region_streets,
    "seconds": round(time.time() - t0),
}
acc = 0; tot = res["streets_total"]
for i, (_, c) in enumerate(street_names.most_common(), 1):
    acc += c
    if acc >= tot / 2:
        res["names_covering_50pct"] = i
        break
json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("DONE", json.dumps({k: res[k] for k in ("streets_total", "streets_unique_names", "cities_unique_names", "names_covering_50pct", "seconds")}, ensure_ascii=False))
