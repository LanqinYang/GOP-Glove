# Prompt for Python Code Refactoring

**Objective:**
Refactor an existing Python script for gesture data preprocessing. The goal is to **replace** the current normalization method (likely `StandardScaler`) with a more robust algorithm called DBAN and to ensure the data augmentation is applied correctly in the pipeline.

**Context:**
- The existing code loads a dataset from a 5-channel sensor glove.
- Each data sample (trial) is a time series of shape `(100, 5)`.
- The core data issues to solve are **baseline drift** and **inconsistent amplitude** between different trials.

---

### **Refactoring Instructions:**

Follow these steps to modify your existing code.

**Step 1: Remove the Old Normalization Method**

- In your current script, find the section where you apply normalization. This is likely a call to `sklearn.preprocessing.StandardScaler` (`scaler.fit_transform(...)` or similar).
- **Delete or comment out this entire code block.** The new DBAN algorithm will completely replace its functionality.

**Step 2: Define the New DBAN Function**

- In a suitable place in your script (e.g., near your other utility functions), define a new Python function named `dban_normalize_trial`.
- **Function's Purpose:** To normalize a *single* data trial.
- **Input:** A NumPy array of shape `(100, 5)`.
- **Output:** A normalized NumPy array of the same shape.
- **Required Logic (to be applied to each of the 5 channels independently):**
    1.  **Detrending:** Remove the baseline drift. Use `scipy.signal.detrend` with `type='linear'`.
    2.  **Amplitude Normalization:**
        a. Calculate a normalization factor. This factor is the **98th percentile** of the **absolute values** of the detrended signal. Use `numpy.percentile`.
        b. Divide the detrended signal by this factor. Remember to handle potential division-by-zero errors if a signal is flat.

**Step 3: Integrate DBAN into Your Data Processing Loop**

- Go to your main data processing loop (this is likely your Leave-One-Subject-Out cross-validation loop).
- Find the point **immediately after** you split your data into `X_train` and `X_test` for a given fold.
- **Modify your code here** to apply the `dban_normalize_trial` function to every sample in both `X_train` and `X_test`.
- *Example modification:*
  ```python
  # Before: You might have had scaler.fit_transform(X_train) here.
  # After:
  X_train_normalized = np.array([dban_normalize_trial(trial) for trial in X_train])
  X_test_normalized = np.array([dban_normalize_trial(trial) for trial in X_test])
  ```

**Step 4: Reposition and Verify Data Augmentation**

- Find your data augmentation logic (`jittering`, `scaling`, etc.).
- **Move this code block** so that it runs **AFTER** Step 3 (after DBAN normalization has been applied).
- **Crucially, verify** that this augmentation logic is applied **ONLY** to the `X_train_normalized` data. The test data (`X_test_normalized`) must **never** be augmented.

**Step 5: Final Logic Check**

- Briefly review your refactored code. For each fold in your LOSO loop, the sequence of operations should now be:
    1. Split raw data into `train` and `test`.
    2. Apply **DBAN** to the `train` set.
    3. Apply **DBAN** to the `test` set.
    4. Apply **Augmentation** *only* to the now-normalized `train` set.
    5. Pass the final processed data to your model for training and evaluation.