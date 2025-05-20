### ✅ **1. Importing Libraries**

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
```

* `pandas` is imported for handling tabular data (`DataFrame`).
* `OneHotEncoder` is imported from `sklearn` to convert categorical variables (like city names) into numerical format that ML models can understand.

---

### ✅ **2. Creating a Sample Data Dictionary**

```python
d = {'sales': [...], 'city': [...], 'size': [...]}
df = pd.DataFrame(data=d)
```

* A dictionary `d` is created with three keys: `'sales'`, `'city'`, and `'size'`.
* Each key maps to a list of values.
* Then, the dictionary is converted into a **Pandas DataFrame** called `df`.

**Example of `df` output:**

| sales   | city    | size   |
| ------- | ------- | ------ |
| 100000  | Tampa   | Small  |
| 222000  | Tampa   | Medium |
| 1000000 | Orlando | Large  |
| ...     | ...     | ...    |

---

### ✅ **3. Setting Up the OneHotEncoder**

```python
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')
```

This line does several things:

* `OneHotEncoder` will convert the `city` column (categorical) into separate binary columns (one for each unique city).
* `handle_unknown='ignore'` ensures that any unseen city during transformation will not cause an error.
* `sparse_output=False` means the output will be a **dense array** (not sparse matrix).
* `.set_output(transform='pandas')` makes the output a **Pandas DataFrame** (instead of NumPy array).

---

### ✅ **4. Applying OneHotEncoder**

```python
ohe_transform = ohe.fit_transform(df[['city']])
```

* This **fits** the encoder on the `'city'` column and then **transforms** it.
* The result is a **new DataFrame** (`ohe_transform`) with one column per unique city, each containing 0 or 1.

**Example of `ohe_transform`:**

| Jacksonville | Miami | Orlando | Tampa |
| ------------ | ----- | ------- | ----- |
| 0            | 0     | 0       | 1     |
| 0            | 0     | 0       | 1     |
| 0            | 0     | 1       | 0     |
| ...          | ...   | ...     | ...   |

---

### ✅ **5. Concatenating Encoded Columns**

```python
df = pd.concat([df, ohe_transform], axis=1).drop(columns=['city'])
```

* This **adds the new one-hot encoded columns** to the original DataFrame.
* Then it **drops** the original `'city'` column, since it's now redundant (already represented by the one-hot columns).

---

### ✅ **6. Printing Final DataFrame**

```python
print(df)
```

* Displays the final DataFrame which now contains:

  * `'sales'`
  * `'size'`
  * One-hot encoded columns: `'Jacksonville'`, `'Miami'`, `'Orlando'`, `'Tampa'`

---

### 🔍 **Summary Example Output:**

| sales   | size   | Jacksonville | Miami | Orlando | Tampa |
| ------- | ------ | ------------ | ----- | ------- | ----- |
| 100000  | Small  | 0            | 0     | 0       | 1     |
| 222000  | Medium | 0            | 0     | 0       | 1     |
| 1000000 | Large  | 0            | 0     | 1       | 0     |
| ...     | ...    | ...          | ...   | ...     | ...   |


