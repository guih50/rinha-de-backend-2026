# Rinha de Backend 2026 — guih50

Minha participação na [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026), tema: detecção de fraudes via busca vetorial k-NN.

**Stack:** Rust · Hyper · Tokio · HAProxy

**Score obtido:** ~5706 / 6000 (p99 ≈ 1.05ms, 3 erros de detecção)

> Esta solução foi desenvolvida com auxílio do [Claude](https://claude.ai) (Anthropic). As decisões de arquitetura, análise de gargalos e otimizações foram feitas iterativamente em pair-programming com o modelo.

---

## O problema

O desafio exige classificar transações como fraude ou legítimas usando busca dos **5 vizinhos mais próximos** (k-NN) em um dataset de referência com **3.000.000 vetores** de 14 dimensões. A pontuação penaliza tanto latência alta (p99 > 1ms) quanto erros de classificação.

O scoring é:
- `score_p99`: até 3000 pts — p99 ≤ 1ms = pontuação máxima
- `score_det`: até 3000 pts — baseado na taxa de erros de detecção
- **Total máximo: 6000 pts**

---

## Abordagem: IVF (Inverted File Index)

Busca bruta em 3M vetores com AVX2 leva ~5ms/query — inviável para p99 ≤ 1ms. A solução foi um índice **IVF** construído em tempo de compilação e embutido no binário via `include_bytes!`.

### Como o IVF funciona

1. **Build-time** (`build.rs`): k-means com 1024 clusters sobre os 3M vetores de referência. Cada vetor é atribuído ao seu centroide mais próximo.
2. **Query-time**: para cada query, calcula distância aos 1024 centroides, seleciona os `NPROBE` clusters mais próximos, e faz busca exaustiva apenas nesses clusters (~3% dos vetores).

Com `NPROBE=27`, o IVF examina em média ~87k vetores por query (vs. 3M no brute-force), caindo de ~5ms para ~0.66ms de computação.

### Quantização i16

Vetores armazenados como `[i16; 16]` com `SCALE=16000`:
- Reduz memória: 3M × 16 × 2 bytes = **96 MB** (vs. 192 MB em f32)
- Permite usar `_mm256_madd_epi16` (SIMD inteiro, mais rápido que float)
- `SCALE=16383` seria o máximo seguro para AVX2 sem overflow, mas usamos 16000 para margem

---

## Otimizações críticas

### 1. HAProxy em modo TCP (maior impacto individual)

O HAProxy original estava configurado em `mode http`. Trocar para `mode tcp` derrubou o p99 de **60ms para 1.5ms** (+1600 pts no score).

Em modo HTTP, o HAProxy faz full HTTP parsing em cada request — inútil quando o backend já fala HTTP/1.1 diretamente. Em modo TCP, ele é apenas um proxy de bytes com zero overhead de parse.

```
mode tcp          # não mode http
tune.bufsize 16384  # CRÍTICO: 8192 causa p99 de 36ms
option splice-auto
```

> **Armadilha:** `tune.bufsize 8192` aumenta o p99 para 36ms. Nunca reduzir abaixo de 16384.

### 2. AVX2 SIMD para distância euclidiana

```rust
// diff[i]^2 somados em pares → 8 × i32
let sq = _mm256_madd_epi16(diff, diff);
// acumulação em i64 para evitar overflow com SCALE=16000
hsum_epi32_to_i64(sq)
```

`_mm256_madd_epi16(diff, diff)` calcula `diff[0]^2 + diff[1]^2` em paralelo para 8 pares, produzindo 8 × i32. Depois, `_mm256_cvtepi32_epi64` promove para i64 antes do hsum, evitando overflow (max par = 32000² × 2 ≈ 2.048B < i32::MAX).

### 3. Prefetch entre clusters

O hardware prefetcher não consegue prever os saltos de ponteiro entre clusters do IVF. Prefetch explícito com `_mm_prefetch(..., _MM_HINT_T1)` reduz stalls de cache na transição entre clusters.

### 4. Warmup dos 96 MB no startup

```rust
// Toca cada página de 4KB antes de aceitar conexões
for i in (0..self.n_vectors).step_by(128) {
    acc = acc.wrapping_add(vectors[i][0] as i32);
}
```

Sem warmup, as primeiras queries sofrem page faults durante o IVF scan. Com warmup, todo o índice está no RAM/cache antes da primeira requisição.

### 5. UDS (Unix Domain Sockets) + volume compartilhado

HAProxy → APIs via Unix Domain Sockets em `/tmp/sockets/`. Evita o overhead do stack TCP (handshake, checksums, etc.) na comunicação interna.

---

## Análise de erros irredutíveis

Com `NPROBE=27`, existem 3 erros que não vale a pena corrigir:

| Erro | Causa | Custo da correção |
|------|-------|-------------------|
| FP@idx=3905 | Fixável com nprobe≥61 | CPU throttling → p99 sobe para 7ms, perde 3× mais em score_p99 do que ganha em score_det |
| FN@idx=5472 | `fc_max=2`, nunca atinge threshold 3 | Irredutível em qualquer nprobe |
| FN@idx=5508 | Fixável com nprobe≥38 | p99 sobe para 1.34ms, perde mais em score_p99 do que ganha |

O ponto ótimo encontrado é `NPROBE=27`.

---

## Estrutura do projeto

```
src/
  main.rs       — servidor Hyper, startup (warmup → bind UDS → accept loop)
  index.rs      — IVF index: from_static_bytes, warmup, query, AVX2 SIMD
  parse.rs      — parser manual de JSON usando memmem (sem serde overhead)
  vectorize.rs  — quantização f64→i16 com SCALE=16000
  responses.rs  — responses pré-construídas como &[u8] estáticos
build.rs        — k-means paralelo, geração do ivf.bin embutido no binário
Dockerfile      — FROM busybox:musl, binário musl estático de ~100MB
haproxy.cfg     — TCP mode, tune.bufsize=16384
docker-compose.yml — 2× API (0.45 CPU, 160MB) + 1× HAProxy (0.10 CPU, 30MB)
```

---

## Como rodar

```bash
# Build do binário musl (requer cross-compiler x86_64-linux-musl-gcc)
cargo build --release --target x86_64-unknown-linux-musl

# Subir o stack
docker compose up --build
```

O build inclui geração do `ivf.bin` (~96MB) via `build.rs`. Espere alguns minutos na primeira vez.
