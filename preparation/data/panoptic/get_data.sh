#!/bin/bash

# ==============================================================================
# CMU Panoptic Dataset - Download & Extract
#
# Usage: bash preparation/data/panoptic/get_data.sh [options]
#
# Options:
#   --all-cameras       Download all CMU camera nodes (requires ~0.5 TB)
#   --snu-endpoint      Use the SNU mirror instead of the CMU server
#   --format <fmt>      Image format for extracted frames (default: jpg)
#   --keep-videos       Keep video files after extracting frames
#   --keep-tars         Keep tar files after extracting poses
# ==============================================================================

set -euo pipefail

# ─── Defaults ─────────────────────────────────────────────────────────────────

FMT="jpg"
KEEP_VIDEOS=false
KEEP_TARS=false
ENDPOINT="http://domedb.perception.cs.cmu.edu"

sequences=(
    "160422_ultimatum1"
    "160224_haggling1"
    "160226_haggling1"
    "161202_haggling1"
    "160906_ian1"
    "160906_ian2"
    "160906_ian3"
    "160906_band1"
    "160906_band2"
    "160906_band3"
    "160906_pizza1"
    "160422_haggling1"
    "160906_ian5"
    "160906_band4"
)

nodes=(3 6 12 13 23) # CMU0 cameras

# ─── Parse arguments ─────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all-cameras)
            nodes=(1 2 3 4 6 7 10 12 13 16 18 19 22 23 30)
            ;;
        --snu-endpoint)
            ENDPOINT="http://vcl.snu.ac.kr/panoptic"
            ;;
        --format)
            FMT="$2"; shift
            ;;
        --keep-videos)
            KEEP_VIDEOS=true
            ;;
        --keep-tars)
            KEEP_TARS=true
            ;;
        *)
            echo "Unknown option: $1"; exit 1
            ;;
    esac
    shift
done

# ─── Prerequisites ────────────────────────────────────────────────────────────

if command -v wget >/dev/null 2>&1; then
    DL="wget -c"
    DL_OUT="-O"
elif command -v curl >/dev/null 2>&1; then
    DL="curl -C -"
    DL_OUT="-o"
else
    echo "Error: wget or curl is required." >&2; exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "Error: ffmpeg is required to extract frames." >&2; exit 1
fi

# ─── Summary ──────────────────────────────────────────────────────────────────

echo "============================================="
echo "  CMU Panoptic Dataset - Download & Extract"
echo "============================================="
echo "  Sequences : ${#sequences[@]}"
echo "  Cameras   : ${nodes[*]}"
echo "  Format    : $FMT"
echo "  Endpoint  : $ENDPOINT"
echo "  Keep videos after extraction: $KEEP_VIDEOS"
echo "  Keep tars after extraction  : $KEEP_TARS"
echo "============================================="
echo ""

DATA_ROOT="./data/panoptic"
TOTAL=${#sequences[@]}
CURRENT=0

for datasetName in "${sequences[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo "[$CURRENT/$TOTAL] ====== $datasetName ======"

    targetDir="$DATA_ROOT/$datasetName"
    mkdir -p "$targetDir"

    # ── 1. Download HD videos ─────────────────────────────────────────────
    mkdir -p "$targetDir/hdVideos"
    panel=0

    for node in "${nodes[@]}"; do
        fileName=$(printf "hd_%02d_%02d.mp4" "$panel" "$node")
        dest="$targetDir/hdVideos/$fileName"

        if [ -f "$dest" ]; then
            echo "  [skip] $fileName already exists"
            continue
        fi

        echo "  [download] $fileName"
        $DL $DL_OUT "$dest" "$ENDPOINT/webdata/dataset/$datasetName/videos/hd_shared_crf20/$fileName" || rm -f "$dest"
    done

    # ── 2. Download calibration ───────────────────────────────────────────
    calibFile="$targetDir/calibration_${datasetName}.json"
    if [ -f "$calibFile" ]; then
        echo "  [skip] calibration already exists"
    else
        echo "  [download] calibration"
        $DL $DL_OUT "$calibFile" "$ENDPOINT/webdata/dataset/$datasetName/calibration_${datasetName}.json" || rm -f "$calibFile"
    fi

    # ── 3. Download & extract 3D poses ────────────────────────────────────
    tarFile="$targetDir/hdPose3d_stage1_coco19.tar"
    poseDir="$targetDir/hdPose3d_stage1_coco19"
    if [ -d "$poseDir" ]; then
        echo "  [skip] 3D poses already extracted"
    else
        if [ ! -f "$tarFile" ]; then
            echo "  [download] 3D poses"
            $DL $DL_OUT "$tarFile" "$ENDPOINT/webdata/dataset/$datasetName/hdPose3d_stage1_coco19.tar" || rm -f "$tarFile"
        fi
        if [ -f "$tarFile" ]; then
            echo "  [extract] 3D poses"
            tar -xf "$tarFile" -C "$targetDir"
            if [ "$KEEP_TARS" = false ]; then
                rm -f "$tarFile"
                echo "  [cleanup] removed $tarFile"
            fi
        fi
    fi

    # ── 4. Extract frames from videos ─────────────────────────────────────
    if [ -d "$targetDir/hdVideos" ]; then
        for node in "${nodes[@]}"; do
            videoFile=$(printf "%s/hdVideos/hd_%02d_%02d.mp4" "$targetDir" 0 "$node")
            outDir=$(printf "%s/hdImgs/%02d_%02d" "$targetDir" 0 "$node")
            imgPattern=$(printf "%s/%02d_%02d_%%08d.%s" "$outDir" 0 "$node" "$FMT")

            if [ ! -f "$videoFile" ]; then
                continue
            fi

            if [ -d "$outDir" ] && [ "$(ls -A "$outDir" 2>/dev/null)" ]; then
                echo "  [skip] frames already extracted for camera $(printf "%02d_%02d" 0 "$node")"
                continue
            fi

            mkdir -p "$outDir"
            echo "  [extract] frames from $(basename "$videoFile")"
            if ! ffmpeg -n -loglevel error -stats -i "$videoFile" -q:v 1 -f image2 -start_number 0 "$imgPattern"; then
                echo "  [warn] ffmpeg failed for $(basename "$videoFile") — deleting and re-downloading on next run"
                rm -rf "$outDir"
                rm -f "$videoFile"
            fi
        done

        # ── 5. Cleanup videos ─────────────────────────────────────────────
        if [ "$KEEP_VIDEOS" = false ]; then
            rm -rf "$targetDir/hdVideos"
            echo "  [cleanup] removed hdVideos/"
        fi
    fi

    echo ""
done

echo "Done. All $TOTAL sequences processed."
