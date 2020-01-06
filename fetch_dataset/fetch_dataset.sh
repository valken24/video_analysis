#!/bin/bash

echo "Downloading..."
wget -O ucf101_rgb.zip.001 http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.001
wget -O ucf101_rgb.zip.002 http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.002
wget -O ucf101_rgb.zip.003 http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.003

echo "Unziping..."
cat ucf101_rgb.zip.001 ucf101_rgb.zip.002 ucf101_rgb.zip.003 > ucf101_rgb.zip && unzip -FF ucf101_rgb.zip

echo "Done."
