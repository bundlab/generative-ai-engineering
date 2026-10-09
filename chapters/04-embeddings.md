# Chapter 04: Vector Space Representations & Dense Embeddings

Dense vector representations (embeddings) form the mathematical foundation of semantic search, Retrieval-Augmented Generation (RAG), and multimodal representation alignment. This chapter covers high-dimensional vector spaces, geometric distance metrics, hyper-dimensional phenomena, and contrastive training objectives like InfoNCE.

---

## 1. Vector Space Representations & Dense vs. Sparse Embeddings

Text representation in machine learning evolved from discrete token counting to continuous latent spaces.


```
              EMBEDDING PARADIGM EVOLUTION


┌─────────────────────────────────┬─────────────────────────────────┐
│        Sparse Vectors           │          Dense Vectors          │
├─────────────────────────────────┼─────────────────────────────────┤
│ • Dimensionality: 10k – 100k+   │ • Dimensionality: 256 – 4,096   │
│ • Values: Binary / Lexical TF   │ • Values: Continuous Real (R)   │
│ • Storage: Term Index Lists     │ • Storage: Dense Floats         │
│ • Exact Keyword Match           │ • Semantic/Conceptual Capture   │
└─────────────────────────────────┴─────────────────────────────────┘

```

### Sparse Representations (Lexical)
Algorithms like BM25 and TF-IDF represent documents as sparse vectors $\mathbf{v} \in \mathbb{R}^{|V|}$, where $|V|$ is the total vocabulary size.
* **Mechanism:** Each dimension corresponds to a specific vocabulary term. Values represent term frequency weighted by inverse document frequency.
* **Limitation:** Fails to capture synonymy, context, or semantic similarity (e.g., "automobile" and "car" share orthogonal vector axes with zero inner product).

### Dense Representations (Semantic)
Deep neural encoders project text into lower-dimensional continuous vector spaces $\mathbf{e} \in \mathbb{R}^d$ (where $d \ll |V|$, typically $d \in [384, 4096]$).
* **Mechanism:** Every dimension carries distributed latent features learned via contrastive training across large text corpora.
* **Property:** Semantically similar concepts map to proximal coordinates within the vector space, enabling semantic match beyond literal keyword overlap.

---

## 2. Geometric Distance Metrics & Similarity Measure

Selecting the correct distance or similarity metric is critical for indexing and searching vector spaces.




```
                  VECTOR METRIC GEOMETRY
  Cosine Similarity                Euclidean (L2) Distance
         u                                  u ──┐
        /                                   │   │ d(u,v)
       / θ                                  │   │
      └───── v                              └───v

```


### Cosine Similarity
Measures the directional alignment (cosine of the angle $\theta$) between two non-zero vectors, invariant to vector magnitude:

$$\text{CosineSim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \frac{\sum_{i=1}^d u_i v_i}{\sqrt{\sum_{i=1}^d u_i^2} \sqrt{\sum_{i=1}^d v_i^2}}$$

* **Range:** $[-1.0, 1.0]$.
* **Usage:** Preferred when document length varies significantly and semantic orientation matters more than norm scale.

### Dot Product (Inner Product / IP)
Computes the unnormalized projection of one vector onto another:

$$\text{DotProduct}(\mathbf{u}, \mathbf{v}) = \mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^d u_i v_i$$

* **Property:** If vectors are $L_2$-normalized ($\|\mathbf{u}\|_2 = \|\mathbf{v}\|_2 = 1.0$), Dot Product is mathematically identical to Cosine Similarity:
  $$\mathbf{u}_{\text{norm}} \cdot \mathbf{v}_{\text{norm}} = \text{CosineSim}(\mathbf{u}, \mathbf{v})$$
* **Performance Note:** Normalized Dot Product allows vector databases to execute dot products directly via SIMD matrix instructions, avoiding floating-point norm divisions during runtime retrieval.

### Euclidean ($L_2$) Distance
Measures the straight-line geometric distance between two points in Euclidean space:

$$d_{L2}(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|_2 = \sqrt{\sum_{i=1}^d (u_i - v_i)^2}$$

* **Monotonic Relationship to Cosine Distance:** For $L_2$-normalized vectors, Euclidean distance directly relates to Cosine Similarity:
  $$d_{L2}^2(\mathbf{u}_{\text{norm}}, \mathbf{v}_{\text{norm}}) = 2 \left(1 - \text{CosineSim}(\mathbf{u}, \mathbf{v})\right)$$

---

## 3. High-Dimensional Geometry & The Curse of Dimensionality

High-dimensional vector spaces ($d \ge 512$) behave in counterintuitive ways that impact search indexes and vector clustering.

### Concentration of Measure
As dimensionality $d \to \infty$, the volume of a sphere concentrates almost entirely in a thin shell near its surface, and the distance between any two randomly chosen vectors approaches a constant relative distance:

$$\lim_{d \to \infty} \frac{\text{Var}(d_{L2}(\mathbf{u}, \mathbf{v}))}{\mathbb{E}[d_{L2}(\mathbf{u}, \mathbf{v})]} = 0$$

* **Impact:** Random vectors in high dimensions become nearly orthogonal to each other ($\mathbf{u} \cdot \mathbf{v} \approx 0$).
* **Practical Implication:** True semantic clustering must pull positive pairs tightly together into localized manifolds to overcome this natural background orthogonality.

### Dimensionality Reduction Trade-Offs
To reduce storage costs and accelerate vector search, techniques like Principal Component Analysis (PCA) or Matryoshka Representation Learning (MRL) slice or project embeddings down to smaller sub-dimensions (e.g., $1536 \to 256$), trading a slight degree of retrieval accuracy for memory efficiency.

---

## 4. Contrastive Training Objectives & InfoNCE Loss

Modern dense embedder models (e.g., MiniLM, bge, E5, OpenAI text-embedding-3) are trained using contrastive learning frameworks to map positive pairs together while pushing negative pairs apart.


```
                  CONTRASTIVE SPACE ALIGNMENT
  Unaligned Space                         Trained Contrastive Space
   ● Neg_1   ○ Pos                         ● Neg_1
      Anchor                                  ● Neg_2    Anchor ○ Pos
   ● Neg_2                                 
```


### InfoNCE Loss (Information Noise-Contrastive Estimation)
InfoNCE frame contrastive learning as a multi-class categorical classification task over one positive pair and $K$ negative pairs.

Given an anchor query $q$, a true positive document $p^+$, and $K$ negative documents $p_i^-$:

$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp\left(\text{sim}(q, p^+) / \tau\right)}{\exp\left(\text{sim}(q, p^+) / \tau\right) + \sum_{i=1}^K \exp\left(\text{sim}(q, p_i^-) / \tau\right)}$$

* **$\text{sim}(\mathbf{u}, \mathbf{v})$:** A similarity score function, typically Cosine Similarity or Normalized Dot Product.
* **$\tau$ (Temperature Parameter):** Controls the hardness of the probability distribution. Lower values of $\tau$ (e.g., $\tau = 0.05$) scale similarity scores, forcing the model to penalize hard negatives severely.

### In-Batch Negatives & Hard Negative Mining
1. **In-Batch Negatives:** Using other positive items within the same mini-batch as implicit negative examples for a given query, scaling $K$ to $B-1$ with minimal memory overhead.
2. **Hard Negative Mining:** Selecting negative documents that share high lexical overlap or top sparse vector scores with the query, but lack true semantic relevance. Training on hard negatives prevents the encoder from relying on surface-level keyword shortcuts.

---

## 5. Summary & Comparison Table

| Metric / Objective | Mathematical Formulation | Primary Strengths | Recommended Deployment Scenario |
| :--- | :--- | :--- | :--- |
| **Cosine Similarity** | $\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$ | Magnitude-agnostic directional matching | Text comparison across varying document lengths |
| **Dot Product (IP)** | $\mathbf{u} \cdot \mathbf{v}$ | Hardware-accelerated matrix multiplication | $L_2$-normalized vectors in production vector DBs |
| **Euclidean ($L_2$)** | $\|\mathbf{u} - \mathbf{v}\|_2$ | Absolute geometric spatial distance | Fixed-scale embeddings and dense spatial clustering |
| **InfoNCE Loss** | $-\log \frac{e^{\text{sim}^+ / \tau}}{\sum e^{\text{sim} / \tau}}$ | Learns structured latent spaces | Fine-tuning dual-encoder embedding models |


---