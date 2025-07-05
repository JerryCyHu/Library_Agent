import csv, json, ast, argparse, pathlib

def parse_line(row):
    
    try:
        _, _, title, author, _, genres_raw, summary = row[:7]
    except ValueError:
        return None
    genres = list(ast.literal_eval(genres_raw).values()) if genres_raw else []
    return {
        "bookname": title.strip(),
        "author"  : author.strip(),
        "genre"   : genres,
        "summary" : summary.strip(),
    }

def main(tsv_path):
    tsv_path = pathlib.Path(tsv_path)
    out_path = tsv_path.with_suffix(".json")
    data = []
    with tsv_path.open(encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            item = parse_line(row)
            if item:
                data.append(item)

    with out_path.open("w", encoding="utf-8") as w:
        json.dump(data, w, ensure_ascii=False, indent=2)

    print(f"Converted {len(data):,} records → {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tsv_file", help="booksummaries.txt path")
    main(parser.parse_args().tsv_file)
