# Different Deployment Strategies in MLOps / Software Engineering

---

## 1. Blue-Green Deployment
➔ Maintain two identical environments: **Blue** and **Green**.

- **Blue** serves production traffic.
- Deploy the new version on **Green**.
- After testing, **switch traffic** from Blue to Green.

### Diagram:
```plaintext
+-----------+      +--------+     +-----------+
|  Users    | ---> | Load    | --> | Blue Env  |
|           |      | Balancer| --> | Green Env |
+-----------+      +--------+     +-----------+
```

## 2. Canary Deployment

➔ **Roll out the new version to a small subset of users first** (like a *canary in a coal mine*).

- **Monitor** stability and performance closely.
- If no issues are detected, **gradually expand** the rollout to more users until it reaches 100%.

### Diagram:
```plaintext
Users --> 90% --> Old Version
        --> 10% --> New Version
(Gradually shifting)
```


## 3. Rolling Deployment

➔ **Update application instances one batch at a time.**

- Some instances serve the old version while others are progressively updated.
- Useful for **zero downtime updates**.

### Diagram:

```plaintext
Instance 1 --> Update
Instance 2 --> Update
Instance 3 --> Update
(Rolling across instances)
```



---

## 4. Shadow Deployment (or Dark Launch)

➔ **New model/version gets live traffic in parallel but outputs are hidden from users.**

- Used mainly for **monitoring and evaluating performance** without impacting user experience.

### Diagram:

```plaintext
+-----------+        +-----------+
|  Users    |----->  | Old Model | (Response sent)
|           |        +-----------+
|           |
|           |        +-----------+
|           |----->  | New Model | (Silent Monitoring)
+-----------+        +-----------+
```

---

## 5. A/B Testing Deployment

➔ **Users are split into two groups:**

- **Group A** uses the Old Version.
- **Group B** uses the New Version.

- Enables **controlled experiments** and **statistical measurement of impact**.

### Diagram:

```plaintext
Group A --> Old Model
Group B --> New Model
(Controlled Experiment)

```

---

## 6. Blue-Green Deployment with Feature Flags

➔ **Combine Blue-Green Deployment with feature toggles:**

- Instead of deploying a completely new environment, **control feature exposure** within the same deployment using feature flags.
- Allows for **gradual feature rollout** and **quick rollback** if needed.

---

## Quick Comparison Table

| Deployment Strategy            | Key Feature                         | Risk Level | Rollback Complexity  |
|---------------------------------|-------------------------------------|------------|-----------------------|
| **Blue-Green**                  | Full environment switch            | Low        | Very Easy             |
| **Canary**                      | Gradual rollout to small % users   | Medium     | Easy                  |
| **Rolling**                     | Batch-wise instance update         | Medium     | Medium                |
| **Shadow**                      | Silent parallel monitoring         | Very Low   | No impact on users    |
| **A/B Testing**                 | Controlled group experimentation   | Medium     | Easy                  |
| **Blue-Green + Feature Flags**  | Controlled feature exposure        | Low        | Very Easy             |

