#!/usr/bin/env bash
# make-dmg.sh — build, Developer ID sign, notarize, and package CoreAIChatMac
# into a notarized .dmg for direct download (no App Store / no sandbox).
#
# Prerequisites:
#   1. A "Developer ID Application" certificate in the login keychain. Create once:
#      Xcode > Settings > Accounts > (team MFN25KNUGJ) > Manage Certificates > + >
#      "Developer ID Application"  (only the account Holder can create it).
#   2. The App Store Connect API key (same one used for the iOS upload) — notarytool uses it.
#
# Usage:
#   (DEVELOPER_DIR: release Xcode 27 (macOS 27 SDK) is the active toolchain — no DEVELOPER_DIR override needed (AB-A-0077, 2026-09-17); export it only to build with a different Xcode)
#   export ASC_KEY_P8=/Users/$USER/Downloads/AuthKey_3ZR8BRVF9H.p8
#   export ASC_KEY_ID=3ZR8BRVF9H
#   export ASC_ISSUER_ID=69a6de96-8f3e-47e3-e053-5b8c7c11a4d1
#   ./make-dmg.sh
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD="$APP_DIR/build"
DD="$BUILD/dd-release"
APP="$DD/Build/Products/Release/CoreAIChatMac.app"
DMG="$BUILD/CoreAIChatMac.dmg"

: "${DEVELOPER_DIR:?set DEVELOPER_DIR to the Xcode 27 beta Developer dir}"
: "${ASC_KEY_P8:?set ASC_KEY_P8 (notarytool)}"; : "${ASC_KEY_ID:?}"; : "${ASC_ISSUER_ID:?}"

ID=$(security find-identity -v -p codesigning | grep "Developer ID Application" | head -1 | awk '{print $2}')
if [ -z "${ID:-}" ]; then
  echo "!!! No 'Developer ID Application' certificate found in the keychain."
  echo "!!! Create one: Xcode > Settings > Accounts > Manage Certificates > + > Developer ID Application."
  exit 1
fi
echo "==> Developer ID identity: $ID"

# 1. Build Release, unsigned (sign explicitly below for full control over hardened runtime).
echo "==> Building Release"
cd "$APP_DIR"
xcodegen generate
rm -rf "$DD"
xcodebuild -project CoreAIChatMac.xcodeproj -scheme CoreAIChatMac -configuration Release \
  -destination 'generic/platform=macOS' -derivedDataPath "$DD" \
  CODE_SIGNING_ALLOWED=NO build

# 2. Sign with Developer ID + hardened runtime + secure timestamp (deep: nested SPM code).
echo "==> Signing"
codesign --force --deep --timestamp --options runtime --sign "$ID" "$APP"
codesign --verify --strict --verbose=2 "$APP"

# 3. Package a compressed DMG and sign it.
echo "==> Building DMG"
rm -f "$DMG"
hdiutil create -volname "CoreAI Zoo for Mac" -srcfolder "$APP" -ov -format UDZO "$DMG"
codesign --force --timestamp --sign "$ID" "$DMG"

# 4. Notarize (waits) and staple the ticket into the DMG.
echo "==> Notarizing (this can take a few minutes)"
xcrun notarytool submit "$DMG" \
  --key "$ASC_KEY_P8" --key-id "$ASC_KEY_ID" --issuer "$ASC_ISSUER_ID" --wait
xcrun stapler staple "$DMG"
xcrun stapler validate "$DMG"

echo "==> Done: $DMG"
echo "    Upload to GitHub Releases and link it from the repo Quickstart."
