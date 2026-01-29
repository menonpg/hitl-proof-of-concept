#!/bin/bash
#
# Master Script: Run Complete HITL Proof-of-Concept
# ==================================================
# Executes all steps of the HITL experiment automatically.
#
# Prerequisites:
#   - ROBOFLOW_API_KEY environment variable set
#   - Python 3.8+
#   - GPU (recommended) or CPU
#
# Usage:
#   bash RUN_ALL.sh
#
# Or with custom device/batch:
#   bash RUN_ALL.sh --device 0 --batch 16
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
DEVICE="0"
BATCH="16"

while [[ $# -gt 0 ]]; do
    case $1 in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --batch)
            BATCH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash RUN_ALL.sh [--device 0] [--batch 16]"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}"
echo "========================================================================"
echo "HITL PROOF-OF-CONCEPT: COMPLETE EXPERIMENTAL PIPELINE"
echo "========================================================================"
echo -e "${NC}"
echo ""
echo "This script will run all 5 steps of the HITL experiment:"
echo "  1. Download dataset from Roboflow"
echo "  2. Split dataset into incremental batches"
echo "  3. Convert to YOLO format"
echo "  4. Train all iterations with transfer learning"
echo "  5. Evaluate and generate visualizations"
echo ""
echo "Settings:"
echo "  Device: $DEVICE"
echo "  Batch size: $BATCH"
echo ""
echo "Estimated time: 2-3 hours on L4 GPU"
echo ""
read -p "Press Enter to start or Ctrl+C to cancel..."
echo ""

# Record start time
START_TIME=$(date +%s)

# Step 1: Download dataset
echo -e "${BLUE}=======================================================================${NC}"
echo -e "${BLUE}Step 1/5: Downloading Dataset${NC}"
echo -e "${BLUE}=======================================================================${NC}"
python 01_download_dataset.py
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Step 1 failed! See error above.${NC}"
    exit 1
fi

# Step 2: Split dataset
echo ""
echo -e "${BLUE}=======================================================================${NC}"
echo -e "${BLUE}Step 2/5: Splitting Dataset${NC}"
echo -e "${BLUE}=======================================================================${NC}"
python 02_split_dataset.py
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Step 2 failed! See error above.${NC}"
    exit 1
fi

# Step 3: Convert to YOLO
echo ""
echo -e "${BLUE}=======================================================================${NC}"
echo -e "${BLUE}Step 3/5: Converting to YOLO Format${NC}"
echo -e "${BLUE}=======================================================================${NC}"
python 03_convert_to_yolo.py
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Step 3 failed! See error above.${NC}"
    exit 1
fi

# Step 4: Train all iterations
echo ""
echo -e "${BLUE}=======================================================================${NC}"
echo -e "${BLUE}Step 4/5: Training All Iterations (This will take 2-3 hours)${NC}"
echo -e "${BLUE}=======================================================================${NC}"
python 04_train_all_iterations.py --device "$DEVICE" --batch "$BATCH"
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Step 4 failed! See error above.${NC}"
    exit 1
fi

# Step 5: Evaluate and plot
echo ""
echo -e "${BLUE}=======================================================================${NC}"
echo -e "${BLUE}Step 5/5: Evaluating and Plotting Results${NC}"
echo -e "${BLUE}=======================================================================${NC}"
python 05_evaluate_and_plot.py
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Step 5 failed! See error above.${NC}"
    exit 1
fi

# Calculate duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

# Final summary
echo ""
echo -e "${GREEN}"
echo "========================================================================"
echo "🎉 HITL PROOF-OF-CONCEPT COMPLETE!"
echo "========================================================================"
echo -e "${NC}"
echo ""
echo "Total time: ${HOURS}h ${MINUTES}m"
echo ""
echo "📁 Results saved to: $(pwd)/results/"
echo ""
echo "📊 Generated files:"
echo "   - results/training_results.json       (Raw metrics)"
echo "   - results/FINAL_REPORT.txt            (Summary report)"
echo "   - results/map50_improvement.png       (Main improvement curve)"
echo "   - results/all_metrics_comparison.png  (All metrics)"
echo "   - results/incremental_improvement.png (Per-iteration gains)"
echo ""
echo "📋 Next steps:"
echo "   1. Review FINAL_REPORT.txt for key findings"
echo "   2. View PNG files for visualizations"
echo "   3. Share results with team"
echo "   4. Use findings to implement real HITL in production"
echo ""
echo -e "${GREEN}✅ Experiment successful!${NC}"
echo ""
