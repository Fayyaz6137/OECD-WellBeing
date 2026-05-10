

def main():
    print("Starting ...")

    import urllib.request, os
    os.makedirs('data/raw', exist_ok=True)

    url = 'https://stats.oecd.org/FileView2.aspx?IDFile=4eb67b25-b4f2-4e2a-b37b-28a9a1a1f4d3'
    print("Downloading OECD Regional Well-Being data...")
    urllib.request.urlretrieve(url, 'data/raw/oecd_regional_wellbeing.xlsx')
    print("Done!")

if __name__ == "__main__":
    main()