@echo off
title VizWiz Full Training Pipeline - Multimodal Assistive Vision
echo ======================================================================
echo    STARTING VIZWIZ FULL DATASET TRAINING (TRAIN, VAL, TEST)
echo ======================================================================
echo.
.\venv\Scripts\python.exe setup_and_train_vizwiz_full.py --epochs 25 --batch 4
echo.
echo Training complete! Check runs\train\vizwiz_full_production\ for weights.
pause
