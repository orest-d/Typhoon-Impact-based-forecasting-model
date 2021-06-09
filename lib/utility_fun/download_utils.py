import requests
import subprocess
import os
from pathlib import Path
import traceback
from tqdm import tqdm

USE_WGET=False

def download_file(url, output_path, cwd=None, use_wget=None):
    """Fetch url and store it in a file specified by output_path.
    The requests library is used by default, can be configured to use wget by seting use_wget parameter to True
    or similarly a global variable USE_WGET. 
    """
    if use_wget is None:
        use_wget = USE_WGET
    print (f"Download {url} to {output_file}")
    folder = Path(output_path).parent
    try:
        if not folder.exists():
            folder.mkdir()
    except:
        traceback.print_exc()

    if use_wget:
        if cwd is None:
            cwd = str(folder)
        output_file = Path(output_path).name
        command = ["wget", url, "-O", output_file]
        print(" ".join(command))
        p = subprocess.call(command ,cwd=cwd)
    else:
        r = requests.get(url, stream=True)
        with open(output_path, 'wb') as fd:
            for chunk in tqdm(r.iter_content(chunk_size=128)):
                fd.write(chunk)

def wgrip2_path():
    import shutil
    pp=shutil.which("wgrib2")
    if pp is None:
        for p in Path(__file__).parents + [Path.cwd()]:
            for pp in (p / "wgrib2", p / "wgrib2.exe"):
                if pp.exists() and pp.is_file():
                    return str(pp)
        raise Exception("The wgrib2 utility is not found. Please install (e.g. use install_wgrib2.sh)")
    else:
        return pp
def convert_grib2_to_netcdf(grib2_paths, netcdf_path):
    wgrib2 = wgrip2_path()
    np = Path(netcdf_path)
    cwd = str(np.parent)
    output_file = np.name
    command = [wgrib2] + grib2_paths + ["-append", "-netcdf"] + output_file
    print(" ".join(command))
    p = subprocess.call(command ,cwd=cwd)
    

if __name__ == "__main__":
    download_file("https://ftp.cpc.ncep.noaa.gov/wd51we/wgrib2/wgrib2.tgz.v3.0.2", "wgrib2.tgz", use_wget=True)