#!/usr/bin/env python3
"""Package the separately built Android SDL host without Gradle or Mac build changes."""
import hashlib, json, os, shutil, subprocess, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SDK=Path.home()/'Library/Android/sdk'
BT=SDK/'build-tools/36.0.0'; JDK=Path('/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home')
CACHE=ROOT.parent/'android/cache/xi-native-android'; OUT=ROOT/'build/android/apk';OUT.mkdir(parents=True,exist_ok=True)
def run(a): subprocess.run(list(map(str,a)),check=True,cwd=ROOT)
java=list((CACHE/'SDL3-3.4.16/android-project/app/src/main/java').rglob('*.java'))+list((ROOT/'android/app/src/main/java').rglob('*.java'))
classes=OUT/'classes';classes.mkdir(exist_ok=True)
run([JDK/'bin/javac','-source','8','-target','8','-bootclasspath',SDK/'platforms/android-36/android.jar','-d',classes]+java)
dex=OUT/'dex';dex.mkdir(exist_ok=True)
run([BT/'d8','--min-api','28','--lib',SDK/'platforms/android-36/android.jar','--output',dex]+list(classes.rglob('*.class')))
unsigned=OUT/'unsigned.apk'
run([BT/'aapt','package','-f','-M',ROOT/'android/app/src/main/AndroidManifest.xml','-I',SDK/'platforms/android-36/android.jar','-F',unsigned])
with zipfile.ZipFile(unsigned,'a',compression=zipfile.ZIP_DEFLATED) as z:
 z.write(dex/'classes.dex','classes.dex')
 z.write(CACHE/'sdl-build/libSDL3.so','lib/arm64-v8a/libSDL3.so')
 main=OUT/'libmain.so'; shutil.copyfile(ROOT/'build/android/libmain.so',main)
 run([Path.home()/'Library/Android/sdk/ndk/27.0.12077973/toolchains/llvm/prebuilt/darwin-x86_64/bin/llvm-strip','--strip-debug',main])
 z.write(main,'lib/arm64-v8a/libmain.so')

ks=CACHE/'test-signing.p12'
if not ks.exists():run([JDK/'bin/keytool','-genkeypair','-keystore',ks,'-storepass','android','-keypass','android','-alias','localtest','-dname','CN=XI Native Local Test','-keyalg','RSA','-validity','3650'])
aligned=OUT/'aligned.apk';run([BT/'zipalign','-f','-P','16','4',unsigned,aligned])
apk=ROOT/'build/android/xi-native-test.apk'
run([BT/'apksigner','sign','--ks',ks,'--ks-pass','pass:android','--ks-key-alias','localtest','--out',apk,aligned])
run([BT/'apksigner','verify',apk])
print(json.dumps({'apk':str(apk),'sha256':hashlib.sha256(apk.read_bytes()).hexdigest(),'bytes':apk.stat().st_size}))
