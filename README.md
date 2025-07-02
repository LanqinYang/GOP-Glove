# GOF Glove

A gesture recognition project using DIY flexible sensors and an Arduino for data acquisition.

## About The Project

This project aims to build a full gesture recognition system, starting with a reliable data collection pipeline. It uses 5 DIY flexible sensors connected to an Arduino to capture finger movements. The data is then streamed to a computer for collection and future model training.

### Built With

*   [Python](https://www.python.org/)
*   [pyserial](https://pyserial.readthedocs.io/)
*   [Arduino](https://www.arduino.cc/)

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

*   An Arduino board (e.g., Nano 33 BLE Sense Rev2) with 5 flexible sensors connected to pins A0-A4.
*   Python 3.8+
*   Arduino IDE

### Installation

1.  **Clone the repo**
    ```sh
    git clone https://github.com/LanqinYang/GOP-Glove.git
    cd your_repository
    ```
2.  **Install Python packages**
    ```sh
    pip install -r requirements.txt
    ```
3.  **Upload the firmware**
    *   Open `arduino/data_collection/sensor_data_collector/sensor_data_collector.ino` in the Arduino IDE.
    *   Select your board and port.
    *   Click "Upload" to flash the firmware to the Arduino.

## Usage

The data collection is handled by `src/data/data_collector.py`, which has two modes.

### 1. Test Mode

This mode is for verifying that the sensors are connected correctly and sending data. It prints the live sensor values to the console and saves them to a CSV file.

1.  Find your Arduino's serial port. On macOS or Linux, you can use:
    ```sh
    ls /dev/cu.usbmodem*
    ```
2.  Run the test script:
    ```sh
    python src/data/data_collector.py test --port /dev/your_arduino_port
    ```

### 2. Auto (Collection) Mode

This is the main mode for collecting gesture data. The script will guide you to collect 10 samples for each of the 10 gestures (0-9).

1.  Run the collection script:
    ```sh
    python src/data/data_collector.py auto --port /dev/your_arduino_port
    ```
2.  Enter a User ID when prompted.
3.  Follow the on-screen instructions, pressing `Enter` to start each 2-second sample recording.

All collected data will be saved as individual `.csv` files in the `datasets/gesture_csv/` directory.

## Roadmap

*   [ ] **Data Preprocessing**: Clean, segment, and normalize the collected raw data.
*   [ ] **Model Training**: Train an initial ML model (e.g., 1D-CNN, Transformer).
*   [ ] **Model Evaluation**: Assess model accuracy and performance.
*   [ ] **On-device Deployment**: Deploy the trained model (TFLite) back to the Arduino for real-time inference.

See the `GOF Glove.md` file for more detailed project notes.

## License

Distributed under the MIT License. See `LICENSE.txt` for more information. 