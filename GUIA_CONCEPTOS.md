# Guía: conceptos y decisiones de diseño

Este documento explica **qué significa cada concepto** con el que nos cruzamos y **por qué tomamos cada decisión**, con nuestros números como ejemplos. Está pensado para leerse de corrido la primera vez y después usarse como consulta. Al final hay un glosario.

Las fórmulas usan esta notación:

| Símbolo | Qué es |
|---|---|
| $s$, $s'$ | el estado ahora y el siguiente (dónde están el pastor y las ovejas) |
| $a$ | la acción (una de las 9: quieto o una de 8 direcciones) |
| $r$ | la recompensa que da el entorno en ese paso |
| $\gamma$ | el descuento (usamos 0,99) |
| $Q(s,a)$ | cuánto espera ganar el agente si en $s$ hace $a$ y después juega bien |
| $\Phi(s)$ | el potencial del estado (sección 4) |

---

## 1. El problema en términos de RL

**Agente y entorno.** El agente es el pastor; el entorno es la simulación de Strömbom (las ovejas reaccionan al pastor y entre ellas). En cada **decisión** el agente mira el estado, elige una acción, y el entorno devuelve el estado nuevo y una **recompensa**.

**Episodio.** Una partida, desde que se ubican las ovejas y el pastor hasta que termina. Puede terminar de dos formas, y la diferencia importa:

- **`terminated`**: terminó *de verdad*. En nuestro caso, solo cuando el centro del rebaño llega a la meta. Después no hay futuro: el valor del estado final es 0.
- **`truncated`**: se cortó por tiempo (`max_pasos`). El futuro *existía*, simplemente dejamos de mirar. Por eso el DQN no debe tratar ese estado como si valiera 0. SB3 lo maneja bien (el buffer guarda cuáles fueron cortes por tiempo). Javier trataba el corte por tiempo como un final con castigo, y eso le enseña al agente algo falso: que el reloj es parte del mundo.

**Retorno.** Lo que el agente intenta maximizar: la suma de recompensas futuras, descontadas,

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots$$

**γ (descuento).** Cuánto vale el futuro comparado con el presente. Con $\gamma = 0{,}99$, una recompensa dentro de 100 decisiones vale $0{,}99^{100} \approx 0{,}37$ de lo que valdría ahora. Una forma útil de pensarlo: **el horizonte efectivo es $1/(1-\gamma)$**, unas 100 decisiones con 0,99 y 20 con 0,95. Más allá de eso el agente casi no "ve". Ese mismo número, $1/(1-\gamma)$, aparece como amplificador de errores en la sección 3.

**Política.** La regla que usa el agente para elegir: dado un estado, qué acción. **Greedy** significa "siempre la acción de mayor $Q$", sin explorar. Las evaluaciones las hacemos greedy.

**Valor Q.** $Q(s,a)$ es el retorno esperado si en $s$ hago $a$ y después sigo jugando bien. Si conociéramos $Q$ exacto, la mejor política sería trivial: elegir $\arg\max_a Q(s,a)$. Todo DQN es aprender $Q$.

---

## 2. DQN, pieza por pieza

### 2.1 Q-learning: aprender $Q$ con la ecuación de Bellman

El valor de hacer $a$ en $s$ es lo que gano ahora más lo que vale el mejor siguiente paso, en promedio. Hace falta el promedio porque el mundo es azaroso: las ovejas tienen ruido y pastorean al azar, así que la misma acción no lleva siempre al mismo $s'$.

$$Q(s,a) = \mathbb{E}\left[\, r + \gamma \max_{a'} Q(s', a') \,\right]$$

Q-learning usa eso como **objetivo** (*target*): después de ver una transición $(s, a, r, s')$, empuja $Q(s,a)$ hacia $y = r + \gamma \max_{a'} Q(s',a')$. Si $s'$ es terminal, $y = r$.

Esto se llama **bootstrapping**: el objetivo se arma con la propia estimación de $Q$. Es lo que hace que el aprendizaje sea rápido, y también lo que propaga los errores (sección 3).

### 2.2 La red neuronal

En vez de una tabla con un $Q$ por estado (imposible: las posiciones son continuas), una red neuronal recibe el estado (8 números) y devuelve 9 valores, uno por acción. La nuestra es un MLP de dos capas de 128. Se entrena por descenso de gradiente para acercar su salida al objetivo $y$. SB3 usa la pérdida de Huber, que es cuadrática para errores chicos y lineal para errores grandes, y recorta el gradiente para que un error grande no rompa la red de un saque.

**Generalización: virtud y problema.** La red comparte parámetros entre estados. Aprender sobre un estado cambia los valores de estados parecidos. Eso permite generalizar a posiciones que nunca vio, pero también permite que entrenar en unos estados **desarme lo que sabía de otros**. Esa es la raíz del olvido catastrófico (sección 6).

### 2.3 La red target

Si el objetivo $y$ se calcula con la misma red que se está entrenando, el blanco se mueve con cada actualización, como querer pegarle a una diana que se corre cuando apuntás. Por eso DQN tiene una **copia retrasada** de la red, la red target, que se usa solo para calcular $y$. Se copia de la principal cada 1.000 decisiones (`target_update_interval`). Entre copias, el blanco queda quieto.

### 2.4 ε-greedy: explorar

Con probabilidad ε el agente elige una acción al azar; si no, la greedy. ε arranca en 1 (todo al azar) y baja linealmente hasta 0,05 durante el primer 30% del entrenamiento (`exploration_fraction`), y después queda en 0,05. Sin exploración el agente nunca descubre que existe algo mejor que lo que ya hace.

Cuando discutimos bajar ε, vimos que los derrumbes ocurren con ε ya en 0,05 y que la evaluación es greedy (ε = 0). **La exploración no era la causa de los derrumbes**: era la red cambiando su política (*policy churn*, sección 6).

### 2.5 Experience replay buffer

**Qué es.** Una memoria donde se guarda cada transición $(s, a, r, s', \text{terminó})$ que el agente vive. Para entrenar, en vez de usar solo el último paso, se toma un **lote al azar** (*batch*, 128 transiciones) de esa memoria. Cuando se llena, la transición más vieja se descarta para hacerle lugar a la nueva, como una cola.

**Para qué sirve:**

1. **Romper la correlación.** Los pasos consecutivos de un episodio son casi iguales. Entrenar con ellos en orden es como estudiar mirando 1.000 veces la misma página: la red se sobreajusta a lo último que vio. Muestrear al azar del buffer mezcla experiencias de muchos momentos y episodios distintos.
2. **Reusar experiencia.** Cada transición se usa en muchas actualizaciones, no una sola vez. Simular cuesta y la experiencia es valiosa.
3. **Mantener la memoria del pasado.** Mientras una transición vieja esté en el buffer, la red sigue entrenándose con ella. El buffer es literalmente lo que la red "recuerda".

**Por qué es posible:** porque DQN es **off-policy**. El objetivo $r + \gamma \max_{a'} Q(s',a')$ no depende de qué política generó la transición: una transición de hace 50.000 decisiones, cuando el agente jugaba distinto, sigue siendo un dato válido sobre cómo funciona el mundo. Un método **on-policy** como PPO aprende sobre la política actual y solo puede usar experiencia que generó esa misma política. Por eso reusa cada tanda de experiencia durante unas pocas pasadas (10 en SB3) mientras la política casi no cambió, la descarta y junta una nueva. No puede tener un buffer de experiencia vieja.

**El tamaño importa (lo que estamos probando ahora).** Con 100.000 transiciones, el buffer guarda las últimas 100.000 decisiones. Cuando el agente ya resuelve el nivel en ~8–10 decisiones por episodio, eso son unos 10.000 episodios **todos exitosos y parecidos**. Las transiciones de cuando el agente se equivocaba (pastor delante de la oveja, alejándose, en una esquina) ya se descartaron. La red deja de ver esos estados y nada ancla sus valores Q ahí (sección 6).

Un buffer de 1.000.000 en un entrenamiento de 250.000 decisiones **no descarta nunca nada**: la red sigue viendo hasta el final los errores del principio. Por eso es una prueba limpia de la hipótesis: si con 1M igual hay derrumbes, la causa no es (solo) que el buffer pierde los estados malos.

**Resultado: igual hubo derrumbes** (2 de 3 semillas, sección 9.1). El buffer grande no fue la solución.

El costo de un buffer grande: la experiencia vieja es de una política vieja y el lote mezcla datos menos relevantes. La red aprende un poco más lento lo nuevo, a cambio de recordar más.

### 2.6 Otros hiperparámetros

| Parámetro | Valor | Qué hace |
|---|---|---|
| `learning_rate` | 5e-4 | Tamaño de cada paso de corrección de la red. Probamos 1e-4: aprendió demasiado lento (56% contra 100%) |
| `learning_starts` | 5.000 | Decisiones al azar antes de empezar a entrenar, para llenar algo el buffer |
| `train_freq` | 4 | Una actualización de la red cada 4 decisiones |
| `batch_size` | 128 | Transiciones por actualización |
| `target_update_interval` | 1.000 | Cada cuánto se copia la red principal a la target |

---

## 3. Sobreestimación y Double DQN

### 3.1 Por qué el `max` sobreestima

La red no conoce $Q$ exacto: cada estimación tiene un error, a veces para arriba y a veces para abajo. El objetivo usa $\max_{a'} Q(s',a')$, y **el máximo de estimaciones ruidosas elige justo las que tienen error para arriba**. En promedio:

$$\mathbb{E}\left[\max_a \hat{Q}(s,a)\right] \;\ge\; \max_a \mathbb{E}\left[\hat{Q}(s,a)\right]$$

Un ejemplo con números: 9 acciones, todas con valor real 0,5, y la red les erra con un error normal de desvío 0,05. El máximo de 9 valores ruidosos da en promedio ~0,575. La red "ve" una acción que vale 15% más de lo que vale cualquiera.

### 3.2 Por qué el bootstrapping lo amplifica

Ese objetivo inflado se usa para entrenar $Q(s,a)$, que a su vez entra en el `max` del estado anterior, que se vuelve a inflar. El error se acumula hacia atrás, y en el peor caso crece del orden de $1/(1-\gamma)$: con $\gamma = 0{,}99$, hasta ~100 veces. Por eso bajar $\gamma$ fue una de las hipótesis (con 0,95 el factor es 20), aunque no resultó la mejor solución.

Hay estados especialmente peligrosos: los que **se alimentan a sí mismos**. Si quedarse en una esquina lleva a la misma esquina, el objetivo de $Q(\text{esquina})$ es $\approx \gamma \max Q(\text{esquina})$. Si se inflan un poco, el error se recicla, y no hay una recompensa real (un final) que lo ancle.

### 3.3 Lo que vimos

En el nivel trivial, con recompensa esparsa (máximo real posible: 1), el DQN común estimaba $Q$ de **1,14 a 1,29** al inicio de los episodios en 2 de 3 semillas, imposible. La tercera daba 0,70–0,74. La diferencia entre la mejor acción y la segunda era de apenas ~0,002–0,013: decenas de veces más chica que el error. La política era casi siempre la misma acción, hacia una esquina, y quedaba **por debajo del azar**.

### 3.4 Double DQN

La causa es que **la misma red elige la acción y la evalúa**: si sobreestima una acción, la elige y además usa ese valor inflado. Double DQN separa los dos roles:

$$\text{DQN:}\quad y = r + \gamma\, Q_{\text{target}}\big(s', \arg\max_{a'} Q_{\text{target}}(s',a')\big)$$

$$\text{Double DQN:}\quad y = r + \gamma\, Q_{\text{target}}\big(s', \arg\max_{a'} Q_{\text{principal}}(s',a')\big)$$

La red principal **elige** la mejor acción y la red target la **evalúa**. Sus errores no están perfectamente correlacionados, así que la acción que la principal sobreestima no necesariamente está sobreestimada en la target, y el sesgo baja. No desaparece: la target es una copia retrasada de la principal, así que sus errores se parecen bastante. En [double_dqn.py](double_dqn.py) son dos líneas distintas respecto del DQN de SB3.

**Por qué ayudó tanto acá:** nuestro problema tenía exactamente el perfil peligroso. Diferencias mínimas entre acciones (en una decisión la oveja se mueve muy poco, así que todas las acciones valen casi lo mismo), γ alto, y estados que se alimentan a sí mismos. En el nivel trivial con 30k decisiones, el DQN común sacó 32% y Double DQN 65%. Con 100k, Double DQN llegó a 90% (y a 100% sumándole la penalización por tiempo).

Double DQN **reduce** la sobreestimación pero no la elimina: en los derrumbes de la fase 2, Q sin shaping volvió a subir a 8–9.

---

## 4. Recompensas, shaping y la ley de Goodhart

### 4.1 Recompensa esparsa

+1 al llegar a la meta y 0 en todo lo demás. Es la tarea pura, imposible de malinterpretar, pero da muy poca información: hasta que el agente llega por casualidad, todas las acciones valen 0 y no hay de dónde aprender. Desde la distancia del paper, el azar nunca llega: 0% en 50 mapas.

### 4.2 Shaping

**Shaping** es agregar recompensas intermedias que guíen al agente ("vas bien, vas mal") antes de que llegue a la meta. Es lo que hace un entrenador de perros: premia los pasos intermedios.

### 4.3 La ley de Goodhart y el farmeo

> "Cuando una medida se vuelve un objetivo, deja de ser una buena medida."

En RL esto se llama *reward hacking* o farmeo: si la recompensa intermedia se puede acumular sin cumplir la tarea, el agente optimiza eso. El agente no "hace trampa": maximiza exactamente lo que le pedimos, y le pedimos otra cosa sin querer.

Ejemplo con tu idea del círculo de compactación, en la versión ingenua: +1 cuando una oveja entra al círculo y −1 cuando sale. Parece balanceado, pero con descuento no lo es. Si una oveja entra en el paso $t$ y sale en el $t+1$:

$$+1 + \gamma \cdot (-1) = 1 - \gamma = 0{,}01 > 0$$

Cada vez que el pastor hace que una oveja entre y salga, gana 0,01. **Entrar y salir para siempre** suma, con descuento, unos 0,5:

$$\sum_{k=0}^{\infty} \gamma^{2k} (1-\gamma) = \frac{1-\gamma}{1-\gamma^2} = \frac{1}{1+\gamma} \approx 0{,}5$$

Llegar a la meta en 125 decisiones vale $\gamma^{125} \approx 0{,}28$. **El agente preferiría farmear el círculo a terminar la tarea.** El problema no es un error de signo: es que el +1 llega antes que el −1, y el descuento hace que lo que llega antes valga más.

### 4.4 Shaping basado en potencial: la solución

Ng, Harada y Russell (1999) demostraron que hay una forma de shaping que **nunca** cambia cuál es la mejor política. Se elige una función del estado, el **potencial** $\Phi(s)$ ("qué tan bien estamos"), y la recompensa extra de cada paso es:

$$F(s, s') = \gamma\, \Phi(s') - \Phi(s)$$

Es decir, "cuánto mejoró el potencial", con el $\gamma$ incluido. Ese $\gamma$ es exactamente lo que hace que funcione.

**El círculo, en versión potencial.** $\Phi$ = cantidad de ovejas dentro del círculo. La oveja entra ($\Phi$ pasa de 0 a 1) y sale ($\Phi$ vuelve a 0):

$$\underbrace{(\gamma \cdot 1 - 0)}_{\text{entra}} + \gamma \cdot \underbrace{(\gamma \cdot 0 - 1)}_{\text{sale}} = \gamma - \gamma = 0$$

**Exactamente cero.** Entrar y salir no da nada, sin importar cuántas veces se repita.

Ojo, que esto es más profundo de lo que parece: **ni siquiera que la oveja entre y se quede da algo neto**. Entrar paga $+\gamma$ ahora, pero cada paso adentro "cuesta" $(\gamma-1)$ (sección 4.5), y en total se devuelve exactamente lo que se cobró. El shaping nunca suma ni resta en el total; **solo cambia cuándo llega la señal**: subir $\Phi$ paga ahora y se devuelve de a poco después. Eso alcanza para guiar, porque el agente aprende antes que "esto va bien", pero no puede convertirse en un objetivo en sí mismo. La suma telescópica lo muestra en general.

**El caso general: la suma telescópica.** Sumemos el shaping de todo un episodio, $s_0, s_1, \dots, s_T$, con descuento:

$$\sum_{t=0}^{T-1} \gamma^t F(s_t, s_{t+1}) = \sum_{t=0}^{T-1} \left( \gamma^{t+1} \Phi(s_{t+1}) - \gamma^t \Phi(s_t) \right)$$

Cada término positivo se cancela con el negativo del paso siguiente ($+\gamma^{1}\Phi(s_1)$ con $-\gamma^1\Phi(s_1)$, y así). Sobreviven solo las puntas:

$$= \gamma^T \Phi(s_T) - \Phi(s_0)$$

Si el episodio termina de verdad (llegó a la meta), definimos $\Phi(\text{terminal}) = 0$, y queda:

$$G_{\text{con shaping}} = G_{\text{original}} - \Phi(s_0)$$

**Esta es la clave.** El shaping total de un episodio no depende de lo que hizo el agente en el medio: solo del estado inicial, que el agente no elige. Para cualquier par de políticas, la diferencia entre sus retornos es la misma con y sin shaping. Lo mismo vale para cada estado y acción:

$$Q_{\text{con shaping}}(s,a) = Q_{\text{original}}(s,a) - \Phi(s)$$

Restar $\Phi(s)$ a todas las acciones de un estado no cambia cuál es la mayor. **La mejor acción en cada estado es la misma con o sin shaping.**

**Por qué no se puede farmear.** Cualquier recorrido que vuelve a un estado $s$ después de $k$ pasos junta, en total, $(\gamma^k - 1)\Phi(s)$. Recorrerlo infinitas veces suma exactamente $-\Phi(s)$, una serie geométrica que converge. Si $\Phi(s) < 0$, ese número es positivo: es el mismo efecto de la sección 4.5, no una ganancia extra. Lo que el agente puede sacarle al shaping en un episodio es $\gamma^T\Phi(s_T) - \Phi(s_0)$. Está **acotado por los valores que puede tomar $\Phi$, contando el 0 del estado terminal** (en valor absoluto, por $2\max|\Phi|$), sin importar lo que haga. El shaping no agrega valor a ninguna estrategia: solo cambia **cuándo** llega la información de "vas bien". Y eso es lo que queremos: que el agente aprenda antes, no que aprenda otra cosa.

**Lo que la teoría no garantiza.** El teorema dice que la política óptima no cambia. No dice que el agente la vaya a encontrar, ni cuánto va a tardar: eso depende de qué tan buena pista es $\Phi$. Un $\Phi$ malo no rompe la tarea, pero puede no ayudar o entorpecer el aprendizaje.

### 4.5 El efecto lateral: $(\gamma - 1)\Phi$

Si el estado no cambia ($s' = s$), el shaping no es 0:

$$F(s, s) = \gamma \Phi(s) - \Phi(s) = (\gamma - 1)\Phi(s)$$

- Si $\Phi(s) < 0$, quedarse quieto **da** un poquito: $(1-\gamma)|\Phi(s)|$.
- Si $\Phi(s) > 0$, quedarse quieto **cuesta** un poquito.

No rompe el teorema (es parte de la suma telescópica), pero cambia lo que el agente ve paso a paso. Nuestro potencial del centro del rebaño es $\Phi_{\text{gcm}} = -d/L$, siempre negativo y más negativo cuanto más lejos. En el escenario del paper (d/L ≈ 1,05), quedarse quieto da **+0,0105** por decisión, y la penalización por tiempo es **−0,0100**: lejos de la meta, esperar sale prácticamente gratis.

Un matiz: lejos de la oveja, todas las acciones dan la misma recompensa inmediata, porque ninguna mueve a la oveja. La penalización por tiempo tampoco distinguía entre acciones ahí. Lo que cambia es el nivel de los valores Q, no la dirección. La variante `potencial_gcm_tiempo_fuerte` (penalización 0,02) prueba si esto pesa en la práctica (sección 9).

### 4.6 El estado final: $\Phi(\text{terminal}) = 0$

Hay que fijar $\Phi$ del estado terminal en 0 para que la suma telescópica cierre. En los cortes por tiempo no se hace, porque el episodio no terminó: el DQN sigue usando el valor del estado.

Consecuencia: el paso que llega a la meta recibe $F = 0 - \Phi(s)$. Con $\Phi_{\text{gcm}}$, como es negativo, ese último paso **suma** un poco: +0,07 con $d \approx 10$ m. Con el potencial de posición (siguiente sección) pasa lo contrario: es positivo, así que al llegar se **pierde**, hasta $w \cdot \Phi_{\text{pos}}$ (0,4–0,8), que se resta del +1 de llegar. No cambia la política óptima, pero conviene tenerlo presente.

### 4.7 Nuestros potenciales

**Centro del rebaño a la meta** ([pastoreo_env.py](pastoreo_env.py), `phi_gcm`):

$$\Phi_{\text{gcm}}(s) = -\frac{\text{distancia(centro del rebaño, meta)}}{L}, \qquad L = 150 \text{ m}$$

Mover la oveja 1 m hacia la meta da ~0,007. Su limitación: si el pastor está a más de 65 m, la oveja no lo siente. Solo pastorea al azar (se mueve con probabilidad 0,05 por paso), así que $\Phi_{\text{gcm}}$ cambia un poco pero sin relación con lo que hace el pastor. **No hay ninguna señal** sobre hacia dónde ir. Esto explica que desde la distancia del paper no aprenda.

**Posición del pastor** (tu idea, `phi_posicion`):

$$\Phi_{\text{pos}}(s) = \underbrace{\frac{1 + \cos\theta}{2}}_{\text{ángulo}} \cdot \underbrace{\frac{D}{D + d}}_{\text{cercanía}}, \qquad D = 50 \text{ m}$$

donde $\theta$ es el ángulo entre (pastor − centro del rebaño) y (centro del rebaño − meta), y $d$ es la distancia del pastor al centro del rebaño.

| Posición del pastor | $\Phi_{\text{pos}}$ |
|---|---|
| Detrás, a 10 m | 0,83 |
| Detrás, a 100 m | 0,33 |
| Al costado, a 10 m | 0,42 |
| Entre el rebaño y la meta | 0 |

Decisiones de diseño:

- **Ángulo y distancia, no solo el ángulo.** Con solo el ángulo, estar perfectamente alineado a 200 m sería "perfecto". El factor de cercanía hace que acercarse sume desde cualquier distancia. La excepción es la línea justo delante del rebaño ($\cos\theta = -1$): ahí $\Phi_{\text{pos}} = 0$ a cualquier distancia, y cerca de esa línea la señal es casi nula. Lo que guía en esa zona es rodear, porque cualquier desvío hacia un costado sube el ángulo.
- **Siempre ≥ 0.** El término de posición no agrega premio por esperar, y achica un poco el de 4.5. Pero no lo elimina: el potencial total $\Phi_{\text{gcm}} + w\,\Phi_{\text{pos}}$ sigue siendo negativo lejos de la meta (en el escenario del paper, ≈ −1,06 + w·0,33 como mucho). Así que quedarse quieto todavía da ~+0,008–0,01 por decisión.
- **Rodear no tiene trampas.** Si el pastor está adelante, rodear al rebaño hace subir $\cos\theta$ de forma continua. El único máximo de $\Phi_{\text{pos}}$ (la "cima" a la que sube el agente) está detrás y pegado al rebaño, así que no hay otro máximo local donde quedar atrapado.
- **Es potencial.** Se suma a $\Phi_{\text{gcm}}$ con un peso $w$ (probamos 0,5 y 1): $\Phi = \Phi_{\text{gcm}} + w \cdot \Phi_{\text{pos}}$. Por la sección 4.4, no se puede farmear. Quedarse detrás del rebaño para siempre no suma nada.
- **El agente puede calcularlo con lo que ya observa** (posición relativa pastor–centro del rebaño y centro del rebaño–meta), así que el potencial depende solo del estado observado.
- **Costo conceptual.** Le pasa al DQN parte de lo que sabe Strömbom ("ponete detrás"). Es válido, pero para comparar DQN contra Strömbom hay que decirlo, y la ablación (con y sin este término) muestra cuánto aporta.
- **Con muchas ovejas.** Si el rebaño está disperso, "detrás del centro" no es lo correcto: Strömbom primero junta, poniéndose detrás de la oveja más lejana. Por eso este término tiene que quedar más débil que el de compactación, como planteaste.

**Los pesos y la jerarquía.** En el shaping por potencial, los pesos no cambian qué es lo óptimo (por 4.4, ningún $\Phi$ lo cambia). Cambian **cuánto empuja cada pista mientras aprende**. "Más débil" significa que la pista de posición no tape a las otras.

### 4.8 Q sin shaping: el diagnóstico

Por 4.4, la red aprende $Q_{\text{con shaping}} = Q_{\text{original}} - \Phi(s)$, y ese número depende de dónde esté la oveja. No sirve para comparar. Sumándole $\Phi(s)$ volvemos a $Q_{\text{original}}$, que es lo que el agente espera ganar en la tarea real. Como la única recompensa positiva es el +1 de llegar, **$Q$ sin shaping nunca debería pasar de 1**. Si pasa, hay sobreestimación. La cota es floja: con la penalización por tiempo, el valor real desde el inicio del nivel trivial es ~0,85. Así que estar por debajo de 1 no prueba que no haya sobreestimación: solo pasarse de 1 la prueba. Es la columna `qmax_sin_shaping` de las curvas, y es lo que nos avisó de los derrumbes antes de que pasaran.

---

## 5. La penalización por tiempo

−0,01 por cada decisión, sumado al shaping. **No es shaping**: cambia la tarea de "llegar" a "llegar rápido", porque una política que tarda el doble ahora vale menos de verdad.

**Por qué es seguro igual.** El peligro de castigar el tiempo es que el agente busque terminar el episodio de otra forma (perder a propósito, si perder terminara el episodio). En nuestro entorno, **la única forma de terminar es ganar**, así que "terminar rápido" y "ganar rápido" son lo mismo.

**Qué hizo.** Quedarse para siempre en un lugar donde no pasa nada (una esquina) pasa a valer $-0{,}01/(1-\gamma) = -1$, en vez de ~0. Eso les quita a esos estados el atractivo que les daba la sobreestimación. Además le da a la red un ancla: el valor real de "no hacer nada" es claramente negativo. Con Double DQN, el nivel trivial pasó de 90% a **100% en las 3 semillas**, en 31–34 pasos (igual que Strömbom), y Q sin shaping quedó en 0,89–0,96. Se mide en el estado en que esté el entrenamiento en ese momento, donde el valor real va de ~0,85 (al inicio) a ~1 (al lado de la meta). Así que esos números son compatibles con no sobreestimar, aunque no lo prueban (sección 4.8). Antes daba 1,1–1,3, imposible.

---

## 6. Inestabilidad: por qué el DQN aprende y después olvida

### 6.1 La tríada mortal

Sutton y Barto llaman así a la combinación de tres cosas:

1. **Aproximación de funciones** (una red neuronal en vez de una tabla),
2. **Bootstrapping** (el objetivo usa la propia estimación),
3. **Aprendizaje off-policy** (aprender de datos de otras políticas).

Cada una sola es inofensiva; juntas pueden hacer que los valores diverjan. DQN tiene las tres. La red target, el replay buffer y Double DQN son parches que lo hacen funcionar en la práctica, pero no hay garantía de convergencia.

### 6.2 Policy churn

Cuando las acciones valen casi lo mismo, un cambio chico en la red (una actualización) cambia cuál es la mayor y la política greedy salta a otra acción. Con diferencias de 0,01 o menos entre las dos mejores acciones, cualquier actualización puede dar vuelta la política en muchos estados a la vez. Por eso vemos curvas que pasan de 100% a 8% y vuelven en 10.000 decisiones.

### 6.3 Olvido catastrófico (*catastrophic forgetting*)

**El concepto general.** Una red neuronal que se entrena con datos nuevos ajusta los mismos parámetros que codificaban lo viejo. Si deja de ver los datos viejos, va sobrescribiendo lo que sabía de ellos. No es un olvido gradual como el humano: puede ser brusco y total.

**Cómo aparece en DQN.** La distribución de los datos cambia **por culpa del propio éxito del agente**:

1. Al principio el agente se equivoca mucho, y el buffer tiene de todo: pastor adelante, alejándose, en esquinas.
2. Cuando aprende, sus episodios pasan a ser cortos y exitosos, y el buffer se llena de eso.
3. Las transiciones de los errores se descartan: el buffer es una cola de las últimas 100.000.
4. La red ya no entrena sobre los estados malos. Sus valores Q ahí quedan sin ancla y, por la sobreestimación, derivan hacia arriba.
5. Alguna acción que lleva a esos estados empieza a parecer buena. La política greedy va hacia allá y el éxito se derrumba.

**La evidencia.** En cada derrumbe de la fase 2, Q sin shaping pasó de 1 (imposible) **antes** de que cayera el éxito, y siguió subiendo hasta 8–9. Por ejemplo, `angulo_30` semilla 2: 1,32 con 100% de éxito, después 2,28 con 88%, y 6%.

**Tu intuición era razonable, y el experimento mostró que no alcanza.** El mecanismo que imaginaste existe. El buffer no "se queda sin espacio" en el sentido de no poder guardar más: al ser una cola, siempre guarda **solo lo más reciente**, y lo más reciente pasa a ser todo igual. Era la primera hipótesis a probar. Pero con 1M y 250k decisiones no se descartó nada, y 2 de 3 semillas igual se derrumbaron (sección 9.1).

Que los datos viejos estén en el buffer no basta. A medida que entra experiencia nueva, los episodios malos del principio pasan a ser una fracción cada vez más chica y casi no aparecen en los lotes de 128. Además, la red puede sobreestimar estados que casi no visita aunque tenga algún dato de ellos. Lo que precede a cada derrumbe es Q sin shaping subiendo por encima de 1: **la sobreestimación sigue siendo la principal sospechosa**, y el olvido sería su consecuencia más que su causa.

### 6.4 Lo que hicimos contra el olvido, y lo que falta probar

- **Mejor checkpoint** (hecho). Guardar el modelo de la mejor evaluación. No arregla el entrenamiento, pero asegura quedarse con un buen modelo. En `angulo_90`, los mejores checkpoints dieron 100% en 100 mapas nuevos en las 3 semillas; los modelos finales, 0/100/0.
- **Buffer más grande** (probado: no lo evita, sección 9.1).
- **Pendiente**, apuntando a la sobreestimación:
  - bajar la tasa de aprendizaje una vez que converge;
  - actualizar la red target más despacio;
  - cortar el entrenamiento cuando Q sin shaping supera ~1,2 durante varias evaluaciones seguidas (*early stopping*).

---

## 7. El entorno: decisiones de diseño

| Decisión | Alternativa | Por qué |
|---|---|---|
| Construir sobre [Pastoreo_flechitas.py](Pastoreo_flechitas.py) (física de Strömbom, meta en el origen) | Reusar el entorno de Javier | Es la pregunta del plan (DQN contra Strömbom), tiene la heurística ya hecha, corre ~2,5 veces más rápido. El de Javier resuelve otra tarea (tranquera, 3 grupos) e itera red y recompensa a la vez |
| API de Gymnasium | Un loop a mano | Permite usar SB3, un DQN probado: si no aprende, el problema es el entorno o la recompensa, no un bug del algoritmo. `check_env` valida la interfaz |
| 9 acciones discretas (quieto + 8 direcciones) | Acción continua | DQN necesita acciones discretas. Medimos que Strömbom discretizado a esas 9 rinde igual (100%), así que no perdemos nada |
| Observación de 8 números (posiciones relativas normalizadas) | Posiciones de cada oveja | Tiene el mismo tamaño con 1 o 30 ovejas y no depende del orden de las ovejas. Tiene lo necesario: dónde está el rebaño respecto del pastor, la meta respecto del rebaño, la oveja más lejana y la dispersión |
| `repetir_accion` (k = 4) | Decidir en cada paso | Cada acción dura 4 pasos de simulación (*frame skip*). Una decisión mueve algo de verdad, el horizonte en decisiones es 4 veces más corto (γ "ve" más lejos en metros) y entrena más rápido |
| Currículum por parámetros (distancia del rebaño, del pastor, ángulo) | Siempre la tarea completa | Desde la distancia del paper no hay señal (azar 0%). Empezamos donde sí se puede aprender y abrimos de a poco |
| `cerco` opcional | Mundo sin bordes siempre | Sin bordes, al principio el pastor aprendía a irse caminando para siempre. Con cerco y 1 oveja aparece la trampa de la esquina (incluso Strömbom cae al 74–77%) |
| Éxito = centro del rebaño a menos de 10 m de la meta | Todas las ovejas adentro | Es el criterio de Strömbom. Con muchas ovejas habrá que sumar "y rebaño compacto", para que no valga un rebaño partido con el centro en la meta |

---

## 8. Cómo medimos: decisiones de método

- **50 mapas fijos** (semillas 20000–20049) para todas las evaluaciones del DQN, sin exploración (greedy). La línea de base de Strömbom ([medir_strombom.py](medir_strombom.py)) usa otros 100 (10000–10099). Así dos corridas se comparan en exactamente las mismas situaciones.
- **Comparar por % de éxito y pasos, nunca por recompensa.** Cada variante de recompensa tiene otra escala; con shaping, la suma de un episodio depende del punto de partida.
- **3 semillas por configuración.** En RL la diferencia entre semillas es enorme (una llega al 100% y otra se queda en 0%). Una sola semilla no permite concluir nada.
- **Siempre contra el azar y contra Strömbom.** El azar dice si el nivel ya se resuelve sin aprender (en el trivial, 36%). Strömbom dice cuál es el techo razonable.
- **Promedio de las últimas evaluaciones**, no el máximo: premia la estabilidad y no un pico suelto.
- **Sesgo de selección del checkpoint.** El mejor checkpoint se elige con los mismos 50 mapas, así que su número es optimista. Por eso lo verificamos en 100 mapas nuevos (semillas 30000–30099), con un chequeo hecho a mano que todavía no está en ningún script.
- **Reproducibilidad.** La misma semilla da exactamente la misma curva (lo confirmamos al repetir `angulo_90`). Un cambio en los resultados es efecto del cambio, no del azar del entrenamiento.
- **Registro de todo.** Cada corrida deja su curva, su configuración (JSON) y una fila en [resultados/registro.csv](resultados/registro.csv). Los escenarios con nombre ([escenarios.yaml](escenarios.yaml)) evitan errores al repetir configuraciones.

---

## 9. Experimento de esta sesión (20/11)

Tres preguntas, con Double DQN y 3 semillas cada una:

1. **¿La posición del pastor como potencial (tu idea) hace aprender desde lejos?** En `rebano_80m` (oveja a 80 m, pastor en cualquier lado a hasta 80 m, sin cerco; Strömbom 100%, azar 4%) y `paper_1_oveja` (azar 0%, la recompensa actual 0%). Variantes: la actual, peso de posición 0,5 y peso 1.
2. **¿Pesa la anulación de la penalización por tiempo (4.5)?** Variante con penalización doble (0,02).
3. **¿Un buffer de 1M evita el olvido?** `angulo_90` con 250k decisiones, donde con 100k las semillas 0 y 2 se derrumbaban (100% → 0%).

### 9.1 Buffer de 1M: no evita el olvido

`angulo_90`, 250k decisiones. "Derrumbe" = pasa de 100% a menos de 20% y no vuelve.

| Semilla | Buffer 100k | Buffer 1M |
|---|---|---|
| s0 | derrumbe a los 170k, termina en 0% | **estable, termina en 100%** |
| s1 | estable, termina en 100% | derrumbe a los 230k, termina en 8% |
| s2 | derrumbe a los 160k, termina en 0% | derrumbe a los 150k, termina en 0% |

Hasta la decisión 100.000 las curvas son **idénticas** con los dos buffers. Tiene sentido: hasta ahí el buffer de 100k todavía no descartó nada y los dos tienen exactamente los mismos datos. Eso hace al experimento muy limpio: las diferencias empiezan justo cuando el de 100k empieza a olvidar.

**Conclusión: la hipótesis "el buffer pierde los estados malos" no es la causa principal.** Con 1M no se descartó ninguna transición, todos los errores del principio seguían ahí, y 2 de 3 semillas igual se derrumbaron. El buffer grande cambió *cuándo* y *a quién* le pasa, no *si* pasa. Que la experiencia vieja esté en el buffer no alcanza: a los 200k decisiones, los episodios malos del principio son una fracción chica del buffer y casi no aparecen en los lotes.

Lo que queda como explicación más probable es la sobreestimación que crece sola (Q sin shaping sube por encima de 1 antes de cada derrumbe) junto con el policy churn. Los próximos candidatos atacan eso directamente:
- bajar la tasa de aprendizaje después de converger;
- actualizar la red target más despacio (cada más decisiones, o de forma suave);
- cortar el entrenamiento cuando Q sin shaping pasa de ~1,2.

Mientras tanto, el mejor checkpoint sigue resolviendo el problema práctico: las 3 semillas con 1M tienen un checkpoint al 100%.

### 9.2 Recompensa de posición del pastor: funciona con peso 0,5

`rebano_80m`: oveja a 80 m, pastor en cualquier lado a hasta 80 m de ella, sin cerco. Strömbom: 100% en 253 pasos; azar: 4%. 300k decisiones.

| Recompensa | Últimas 3 evals (s0 / s1 / s2) | Promedio | Mejor checkpoint en 100 mapas nuevos | Pasos |
|---|---|---|---|---|
| Actual (`potencial_gcm_tiempo`) | 7 / 3 / 17 | 9% | 87 / 29 / 22 | 198–260 |
| Penalización doble (`_fuerte`) | 96 / 44 / 37 | 59% | 100 / 62 / 57 | 174–189 |
| **Posición, peso 0,5** | **39 / 96 / 69** | **68%** | **80 / 95 / 99** | **151–189** |
| Posición, peso 1 | 9 / 1 / 10 | 7% | 44 / 4 / 82 | 172–309 |

- **Tu idea funciona, con el peso justo.** Con peso 0,5, los mejores checkpoints resuelven 80–99% de mapas que nunca vieron, contra 22–87% de la recompensa actual. La semilla 2 todavía estaba subiendo al final (72% → 98%): con más entrenamiento probablemente mejora.
- **Con peso 1 es peor que no usarla.** Una explicación posible es la de la sección 4.6: con $\Phi_{\text{pos}}$ positivo, el paso que llega a la meta pierde hasta $w \cdot \Phi_{\text{pos}} \approx 0{,}8$ del +1 de llegar. Con $w = 1$, llegar casi no paga en el último paso, y la pista "estar detrás" tapa la de "llegar". Coincide con tu intuición de que la señal de posición tiene que ser más débil que las otras.
- **La penalización doble también ayuda mucho** (de 9% a 59%). Eso apoya que la anulación de la sección 4.5 pesaba de verdad: a 80 m, $(1-\gamma)|\Phi_{\text{gcm}}| \approx 0{,}005$ se comía la mitad de la penalización.
- **Los mejores modelos son más rápidos que Strömbom** (151–189 pasos contra 253). Strömbom es una regla general; el DQN optimiza exactamente esta tarea, con una penalización que premia la velocidad.

### 9.3 Escenario del paper: primeros éxitos, todavía inestables

`paper_1_oveja` (~150 m, sin cerco), 300k decisiones. Antes: 0% en todas las evaluaciones, incluso con 500k.

| Recompensa | Mejor evaluación (s0 / s1 / s2) | Final |
|---|---|---|
| Penalización doble | 2 / 0 / 0 | 0 / 0 / 0 |
| Posición, peso 0,5 | 12 / 0 / 0 | 0 / 0 / 0 |
| Posición, peso 1 | 4 / 24 / 42 | 0 / 6 / 0 |

Con peso 1 aparecen los **primeros éxitos desde la distancia del paper**: el mejor checkpoint de la semilla 2 resuelve 45% de mapas nuevos, en 597 pasos (Strömbom: ~530). Pero es inestable y se pierde.

Desde tan lejos todavía cuesta encontrar la meta explorando. El siguiente paso natural es un **currículum**: arrancar desde un modelo que ya resuelve `rebano_80m` y seguir entrenando en el paper, en vez de empezar de cero. También falta probar **posición 0,5 + penalización doble juntas**: cada una ayudó por separado en `rebano_80m`.

---

## 10. Glosario

| Término | Significado |
|---|---|
| **Acción greedy** | La de mayor $Q$ en el estado actual |
| **Batch (lote)** | Grupo de transiciones del buffer que se usa en una actualización de la red |
| **Bootstrapping** | Armar el objetivo de aprendizaje con la propia estimación ($r + \gamma \max Q(s')$) |
| **Catastrophic forgetting** | La red pierde lo aprendido sobre datos que dejó de ver (sección 6.3) |
| **Checkpoint** | Copia guardada del modelo en un momento del entrenamiento |
| **Currículum** | Entrenar de lo fácil a lo difícil |
| **Deadly triad** | Aproximación + bootstrapping + off-policy: la combinación que puede divergir |
| **Double DQN** | La red principal elige la acción del objetivo y la target la evalúa (sección 3.4) |
| **DQN** | Q-learning con una red neuronal, replay buffer y red target |
| **Episodio** | Una partida completa |
| **ε-greedy** | Con probabilidad ε, acción al azar; si no, la greedy |
| **Frame skip** | Repetir cada acción varios pasos de simulación (`repetir_accion`) |
| **γ (descuento)** | Cuánto vale el futuro respecto del presente; horizonte ≈ $1/(1-\gamma)$ |
| **GCM** | Centro de masa del rebaño (*global center of mass*), el promedio de las posiciones |
| **Goodhart (ley de)** | Una medida que se vuelve objetivo deja de medir bien; en RL, *reward hacking* |
| **Huber (pérdida)** | Cuadrática para errores chicos y lineal para grandes; más robusta que la cuadrática |
| **Off-policy** | Aprende de datos generados por cualquier política (DQN). Permite el replay buffer |
| **On-policy** | Aprende solo de datos de la política actual (PPO) |
| **Policy churn** | La política greedy cambia mucho con actualizaciones chicas, porque las acciones valen casi lo mismo |
| **Potencial ($\Phi$)** | Función del estado que mide "qué tan bien estamos"; el shaping es $\gamma\Phi(s') - \Phi(s)$ |
| **Q (valor)** | Retorno esperado de hacer $a$ en $s$ y después jugar bien |
| **Q sin shaping** | $Q + \Phi(s)$: lo que el agente espera ganar en la tarea real; no debería pasar de 1 |
| **Recompensa esparsa** | Solo al final (+1 al llegar) |
| **Red target** | Copia retrasada de la red, usada para calcular el objetivo |
| **Replay buffer** | Memoria de transiciones de la que se muestrea al azar para entrenar |
| **Retorno** | Suma descontada de recompensas futuras |
| **Shaping** | Recompensas intermedias para guiar el aprendizaje |
| **Sobreestimación** | El `max` sobre estimaciones ruidosas da valores sistemáticamente altos |
| **Terminated / truncated** | Fin real del episodio contra corte por tiempo |
| **Transición** | Un paso guardado: $(s, a, r, s', \text{terminó})$ |
