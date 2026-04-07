
# AI-Powered Automated Waste Segregation and E-Waste Recycling System

## 📌 Project Overview
The AI-Powered Automated Waste Segregation and E-Waste Recycling System bridges the gap between simple visual object recognition and intelligent, end-to-end waste management. Traditional recycling systems often operate as "black boxes" that ignore waste contamination, fail to map internal device hazards, rely on static processing parameters, and lack real-time economic forecasting. 

This project transitions waste handling from a linear "Extract-Consume-Dispose" sequence to a circular "value recovery" model. By combining computer vision, materials science, and time-series forecasting, this system is designed to achieve a **100% landfill diversion rate** (Zero-Waste).

---

## ⚙️ System Architecture (The Four Components)
This repository hosts a microservices-based web application divided into four core intelligent components.

### **Component 01: Smart Contamination Detection System**
*   **Focus:** Moves beyond identifying *what* the waste is to evaluating its *condition*.
*   **Mechanism:** Uses a ResNet50-based Convolutional Neural Network (CNN) integrated with Gray-Level Co-occurrence Matrix (GLCM) features to perform surface texture analysis. 
*   **Output:** Detects grease, rust, and mud to assign a three-tier quality grade (Grade A: Clean, Grade B: Washable, Grade C: Landfill/Energy Recovery).

### **Component 02: Intelligent Hazard and Material Identification System**
*   **Focus:** Provides device-to-material mapping and explainable hazard reasoning for e-waste.
*   **Mechanism:** Utilizes YOLOv8 for rapid device category detection (e.g., smartphones, laptops) and links the identified models to a structured materials knowledge base built from iFixit and PubChem safety data.
*   **Output:** Generates transparent cause-and-effect hazard explanations (e.g., thermal runaway risks of lithium-ion batteries) alongside step-by-step safety handling instructions and PPE recommendations.

### **Component 03: Smart Process Optimization Engine**
*   **Focus:** Replaces static recycling parameters with a dynamic, weather-resilient "Dual-Pathway Adaptive Optimization Engine".
*   **Mechanism:** Automatically distinguishes between Abiotic (Plastics/Metals) and Biotic (Organics) streams. It applies real-time moisture sensing to adjust processing logic.
*   **Output:** For wet plastics, it triggers pre-drying cycles; for wet organics, it calculates C:N ratios and recommends biological additives to accelerate decomposition without harmful chemicals.

### **Component 04: Predictive Economic Valuation & Strategic Disposition Dashboard**
*   **Focus:** Solves the "Management Dead-End" by providing strategic financial and zero-waste disposition logic.
*   **Mechanism:** Implements an ARIMA (Auto-Regressive Integrated Moving Average) time-series model to forecast commodity prices (Gold, Copper, Lithium) up to 90 days in advance.
*   **Output:** Recommends "Sell/Hold" strategies for high-value metals and automatically routes non-recyclable "Grade C" residuals to thermal energy recovery or pyrolysis to ensure 0% landfill contribution.

---

## 🛠️ Technology Stack
The system is built using a modern, scalable microservices architecture.

*   **Frontend:** React.js, Tailwind CSS (High-contrast circular economy UI), Recharts (for market trends).
*   **Backend:** Python, FastAPI (Four independent microservices for each component API).
*   **AI/ML Models:** YOLOv8 (Object Detection), ResNet50 (CNN), ARIMA & LSTM (Time-series forecasting), Decision Tree Classifiers.
*   **Databases:** PostgreSQL (Historical price storage and digital manifests), Neo4j (Graph database for materials knowledge mapping).

---

## 📊 Datasets Used
The models are trained and validated using a multi-domain data architecture:
*   **Visual Data:** TrashNet, TACO, RealWaste, and a Local E-Waste Collection dataset.
*   **Materials & Safety Data:** iFixit API (10,000+ device teardowns), PubChem Safety Data Sheets (200+ materials).
*   **Economic & Processing Data:** Kaggle Metals (10 years of historical pricing), World Bank Pink Sheet, ScrapMonster, EPA WARM Models, and IEA Biogas Yield Benchmarks.

---


## 🤝 Project Team & Component Allocation
*   **Component 01:** Smart Contamination Detection System
*   **Component 02:** Intelligent Hazard and Material Identification System 
*   **Component 03:** Smart Process Optimization Engine
*   **Component 04:** Predictive Economic Valuation & Strategic Disposition Dashboard
```

