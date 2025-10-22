import numpy as np
import random
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# ====================================================================
# CONFIGURAÇÃO DE SEED PARA REPRODUTIBILIDADE
# ====================================================================
SEED_GLOBAL = 42
np.random.seed(SEED_GLOBAL)
random.seed(SEED_GLOBAL)

# ====================================================================
# PARTE A: PARÂMETROS E GERAÇÃO DE DADOS SINTÉTICOS
# ====================================================================

PONTUACAO_RENDA = {'1': 10, '2': 25, '3': 40}
FATOR_EMPREGO = {'1': 1.0, '2': 2.0}
PONTUACAO_HISTORICO = {'1': 0, '2': 50, '3': -20}
PONTUACAO_TEMPO = {'1': 0, '2': 10, '3': 20}
LIMITE_MAX_NOVO_CLIENTE = 1300.00

# Carteira global fixa (gerada uma única vez)
_CARTEIRA_CACHE = None

def calcular_score_inicial(renda_cat_key, tipo_emprego_key, historico_key, tempo_loja_key):
    """Cálculo tradicional do score (mantido para compatibilidade)"""
    pr = PONTUACAO_RENDA.get(renda_cat_key, 0)
    f = FATOR_EMPREGO.get(tipo_emprego_key, 1.0)
    prp = pr * f
    ph = PONTUACAO_HISTORICO.get(historico_key, 0)
    pt = PONTUACAO_TEMPO.get(tempo_loja_key, 0)
    pa = ph + pt
    return prp + pa

def gerar_carteira_sintetica_avancada(tamanho=2000):
    """Gera dados sintéticos com features mais complexas"""
    dados = []
    categorias_renda = list(PONTUACAO_RENDA.keys())
    tipos_emprego = list(FATOR_EMPREGO.keys())
    historicos = list(PONTUACAO_HISTORICO.keys())
    tempos = list(PONTUACAO_TEMPO.keys())

    for _ in range(tamanho):
        renda_cat = random.choice(categorias_renda)
        tipo_emprego = random.choice(tipos_emprego)
        historico = random.choice(historicos)
        tempo_loja = random.choice(tempos)

        # Features numéricas
        renda_cat_num = int(renda_cat)
        tipo_emprego_num = int(tipo_emprego)
        historico_num = int(historico)
        tempo_loja_num = int(tempo_loja)
        
        score_base = calcular_score_inicial(renda_cat, tipo_emprego, historico, tempo_loja)
        
        # Features derivadas (engenharia de features)
        estabilidade = renda_cat_num * tipo_emprego_num
        experiencia = tempo_loja_num * (1 if historico_num == 2 else 0.5)
        
        valor_compra = random.randint(100, 3000)
        racio_inicial = score_base / valor_compra if valor_compra > 0 else 0
        
        # Taxa de juros variável baseada no perfil
        juros_base = 0.10
        if historico_num == 2 and tipo_emprego_num == 2:
            juros = juros_base * 0.8  # Desconto para bons clientes
        elif historico_num == 3:
            juros = juros_base * 1.5  # Penalização para maus pagadores
        else:
            juros = juros_base
        
        # Risco mais sofisticado
        risco_base = max(0.01, min(0.9, (120 - score_base) / 100))
        
        # Ajuste de risco baseado em múltiplos fatores
        if renda_cat_num == 3 and tipo_emprego_num == 2:
            risco_ajustado = risco_base * 0.7
        elif historico_num == 3:
            risco_ajustado = min(0.9, risco_base * 1.3)
        else:
            risco_ajustado = risco_base
            
        # Simula resultado real (pago ou não pago)
        pagou = random.random() > risco_ajustado
        
        dados.append({
            'features': [renda_cat_num, tipo_emprego_num, historico_num, 
                        tempo_loja_num, estabilidade, experiencia, racio_inicial],
            'score_base': score_base,
            'valor_compra': valor_compra,
            'juros': juros,
            'juros_esperado': valor_compra * juros,
            'risco_perda_percentual': risco_ajustado,
            'pagou': pagou
        })
    return dados

# ====================================================================
# PARTE B: REDE NEURAL PARA PREDIÇÃO DE SCORE DINÂMICO
# ====================================================================

class ModeloScoreDinamico:
    """Rede Neural para predição de score baseada em múltiplas features"""
    
    def __init__(self):
        self.modelo = MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=500,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.treinado = False
        
    def treinar(self, dados_carteira):
        """Treina a rede neural com os dados da carteira"""
        X = np.array([d['features'] for d in dados_carteira])
        y = np.array([d['score_base'] for d in dados_carteira])
        
        X_scaled = self.scaler.fit_transform(X)
        self.modelo.fit(X_scaled, y)
        self.treinado = True
        
        # Validação cruzada
        scores = cross_val_score(self.modelo, X_scaled, y, cv=5, 
                                scoring='neg_mean_squared_error')
        mse = -scores.mean()
        return mse
    
    def prever_score(self, features):
        """Prediz o score para um conjunto de features"""
        if not self.treinado:
            raise ValueError("Modelo precisa ser treinado primeiro!")
        
        features_scaled = self.scaler.transform([features])
        return self.modelo.predict(features_scaled)[0]

# ====================================================================
# PARTE C: OTIMIZAÇÃO HÍBRIDA (AG + PSO)
# ====================================================================

class ParticleSwarmOptimizer:
    """PSO para otimização complementar ao AG"""
    
    def __init__(self, n_particulas=30, n_iteracoes=50):
        self.n_particulas = n_particulas
        self.n_iteracoes = n_iteracoes
        self.w = 0.7  # Inércia
        self.c1 = 1.5  # Coeficiente cognitivo
        self.c2 = 1.5  # Coeficiente social
        
    def otimizar(self, funcao_fitness, limite_min=0.001, limite_max=2.0):
        """Executa PSO para encontrar melhor coeficiente C"""
        # Inicialização
        posicoes = np.random.uniform(limite_min, limite_max, self.n_particulas)
        velocidades = np.random.uniform(-0.1, 0.1, self.n_particulas)
        
        pbest_posicoes = posicoes.copy()
        pbest_fitness = np.array([funcao_fitness(p) for p in posicoes])
        
        gbest_idx = np.argmax(pbest_fitness)
        gbest_posicao = pbest_posicoes[gbest_idx]
        gbest_fitness = pbest_fitness[gbest_idx]
        
        # Iterações
        for _ in range(self.n_iteracoes):
            for i in range(self.n_particulas):
                r1, r2 = random.random(), random.random()
                
                velocidades[i] = (self.w * velocidades[i] + 
                                 self.c1 * r1 * (pbest_posicoes[i] - posicoes[i]) +
                                 self.c2 * r2 * (gbest_posicao - posicoes[i]))
                
                posicoes[i] += velocidades[i]
                posicoes[i] = np.clip(posicoes[i], limite_min, limite_max)
                
                fitness_atual = funcao_fitness(posicoes[i])
                
                if fitness_atual > pbest_fitness[i]:
                    pbest_posicoes[i] = posicoes[i]
                    pbest_fitness[i] = fitness_atual
                    
                    if fitness_atual > gbest_fitness:
                        gbest_posicao = posicoes[i]
                        gbest_fitness = fitness_atual
        
        return gbest_posicao, gbest_fitness

# ====================================================================
# PARTE D: ALGORITMO GENÉTICO MULTI-OBJETIVO AVANÇADO
# ====================================================================

TAMANHO_POPULACAO = 100
NUM_GERACOES = 100
TAXA_MUTACAO = 0.05
TAXA_CROSSOVER = 0.8
LIMITE_C_MIN = 0.001
LIMITE_C_MAX = 2.0

DADOS_CARTEIRA = gerar_carteira_sintetica_avancada(tamanho=2000)

def funcao_aptidao_multiobjetivo(coeficiente_c, peso_lucro=0.7, peso_risco=0.3):
    """
    Função fitness multi-objetivo:
    - Maximiza lucro líquido esperado
    - Minimiza exposição ao risco
    """
    lucro_total = 0
    risco_total = 0
    aprovacoes = 0
    FATOR_AGRESSIVIDADE = 0.30
    
    for cliente in DADOS_CARTEIRA:
        score = cliente['score_base']
        valor_compra = cliente['valor_compra']
        
        if (score / valor_compra) >= coeficiente_c:
            aprovacoes += 1
            chance_pagar = 1 - cliente['risco_perda_percentual']
            
            ganho_esperado = cliente['juros_esperado'] * chance_pagar
            perda_esperada = valor_compra * cliente['risco_perda_percentual']
            
            bonus_agressividade = valor_compra * FATOR_AGRESSIVIDADE
            
            lucro_na_operacao = ganho_esperado - perda_esperada + bonus_agressividade
            lucro_total += lucro_na_operacao
            
            # Penaliza exposição a alto risco
            risco_total += cliente['risco_perda_percentual'] * valor_compra
    
    # Penaliza se aprovar muito pouco (< 30%) ou muito (> 90%)
    taxa_aprovacao = aprovacoes / len(DADOS_CARTEIRA)
    penalidade_aprovacao = 0
    if taxa_aprovacao < 0.3:
        penalidade_aprovacao = (0.3 - taxa_aprovacao) * 10000
    elif taxa_aprovacao > 0.9:
        penalidade_aprovacao = (taxa_aprovacao - 0.9) * 5000
    
    # Fitness combinado
    fitness = (peso_lucro * lucro_total - 
               peso_risco * risco_total - 
               penalidade_aprovacao)
    
    return fitness

def criar_individuo():
    return random.uniform(LIMITE_C_MIN, LIMITE_C_MAX)

def selecao_torneio(populacao, aptidoes, k, tamanho_torneio=5):
    """Seleção por torneio com tamanho configurável"""
    selecionados = []
    candidatos = list(zip(aptidoes, populacao))
    for _ in range(k):
        torneio = random.sample(candidatos, tamanho_torneio)
        melhor = max(torneio, key=lambda x: x[0])
        selecionados.append(melhor[1])
    return selecionados

def crossover_blend(pai1, pai2, alpha=0.5):
    """BLX-α crossover para melhor exploração"""
    if random.random() < TAXA_CROSSOVER:
        d = abs(pai1 - pai2)
        min_val = min(pai1, pai2) - alpha * d
        max_val = max(pai1, pai2) + alpha * d
        return random.uniform(max(min_val, LIMITE_C_MIN), 
                            min(max_val, LIMITE_C_MAX))
    return random.choice([pai1, pai2])

def mutacao_gaussiana(individuo, sigma=0.05):
    """Mutação gaussiana adaptativa"""
    if random.random() < TAXA_MUTACAO:
        desvio = np.random.normal(0, sigma)
        individuo += desvio
        return max(LIMITE_C_MIN, min(LIMITE_C_MAX, individuo))
    return individuo

def rodar_ag_avancado():
    """AG com elitismo e diversidade populacional"""
    populacao = [criar_individuo() for _ in range(TAMANHO_POPULACAO)]
    melhor_c_global = None
    melhor_aptidao_global = -np.inf
    historico_fitness = []

    for geracao in range(NUM_GERACOES):
        aptidoes = [funcao_aptidao_multiobjetivo(c) for c in populacao]
        melhor_aptidao_atual = max(aptidoes)
        melhor_c_atual = populacao[aptidoes.index(melhor_aptidao_atual)]
        
        historico_fitness.append(melhor_aptidao_atual)

        if melhor_aptidao_atual > melhor_aptidao_global:
            melhor_aptidao_global = melhor_aptidao_atual
            melhor_c_global = melhor_c_atual

        # Elitismo: mantém os 10% melhores
        elite_size = TAMANHO_POPULACAO // 10
        indices_elite = np.argsort(aptidoes)[-elite_size:]
        nova_populacao = [populacao[i] for i in indices_elite]

        # Seleção e reprodução
        pais = selecao_torneio(populacao, aptidoes, TAMANHO_POPULACAO // 2)

        while len(nova_populacao) < TAMANHO_POPULACAO:
            pai1, pai2 = random.choices(pais, k=2)
            filho = crossover_blend(pai1, pai2)
            
            # Mutação adaptativa: aumenta no final
            sigma = 0.05 * (1 + geracao / NUM_GERACOES)
            filho_mutado = mutacao_gaussiana(filho, sigma)
            nova_populacao.append(filho_mutado)

        populacao = nova_populacao
        
        # Diversidade: substitui 5% piores por novos aleatórios
        if geracao % 20 == 0:
            n_substituir = TAMANHO_POPULACAO // 20
            for i in range(n_substituir):
                populacao[i] = criar_individuo()

    return melhor_c_global, historico_fitness

def otimizacao_hibrida():
    """Combina AG e PSO para melhor resultado"""
    print("🧬 Fase 1: Algoritmo Genético...")
    melhor_c_ag, hist_ag = rodar_ag_avancado()
    
    print("🌊 Fase 2: Particle Swarm Optimization...")
    pso = ParticleSwarmOptimizer(n_particulas=30, n_iteracoes=50)
    melhor_c_pso, fitness_pso = pso.otimizar(funcao_aptidao_multiobjetivo)
    
    print("\n📊 Resultados da Otimização Híbrida:")
    print(f"   AG  → C = {melhor_c_ag:.4f} | Fitness = {funcao_aptidao_multiobjetivo(melhor_c_ag):.2f}")
    print(f"   PSO → C = {melhor_c_pso:.4f} | Fitness = {fitness_pso:.2f}")
    
    # Retorna o melhor entre AG e PSO
    if fitness_pso > funcao_aptidao_multiobjetivo(melhor_c_ag):
        print(f"✅ PSO venceu! Melhor C = {melhor_c_pso:.4f}")
        return melhor_c_pso
    else:
        print(f"✅ AG venceu! Melhor C = {melhor_c_ag:.4f}")
        return melhor_c_ag

# ====================================================================
# EXPORTAÇÕES
# ====================================================================

def treinar_sistema_completo():
    """Treina RNA e otimiza coeficiente C"""
    print("=" * 60)
    print("  SISTEMA AVANÇADO DE CRÉDITO - TREINAMENTO")
    print("=" * 60)
    
    # 1. Treina Rede Neural
    print("\n🤖 Treinando Rede Neural para Score Dinâmico...")
    modelo_rna = ModeloScoreDinamico()
    mse = modelo_rna.treinar(DADOS_CARTEIRA)
    print(f"   MSE na validação cruzada: {mse:.2f}")
    
    # 2. Otimização Híbrida
    print("\n⚙️ Otimizando Política de Crédito (AG + PSO)...")
    melhor_c = otimizacao_hibrida()
    
    print("\n" + "=" * 60)
    print(f"🎯 SISTEMA TREINADO COM SUCESSO!")
    print(f"   Coeficiente C Ótimo: {melhor_c:.4f}")
    print("=" * 60)
    
    return melhor_c, modelo_rna