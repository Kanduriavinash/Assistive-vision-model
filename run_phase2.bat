@echo off
title Multimodal Assistive Vision System - Phase 2 Master Pipeline
echo ======================================================================
echo    MULTIMODAL ASSISTIVE VISION SYSTEM (PHASE 2 - 100%% EVALUATION)
echo ======================================================================
echo.
echo Running End-to-End Multimodal Spatial Pipeline...
echo YOLOv8 + Depth Anything V2 + EasyOCR + BLIP VLM + Spatial Fusion
echo.
.\venv\Scripts\python.exe main_phase2.py --dir test_images --output output\phase2_eval
echo.
echo ======================================================================
echo Phase 2 Evaluation Completed! Check output\phase2_eval\ for results.
echo ======================================================================
pause
