import numpy as np
import random

# ====================================================================
# PARTE A: PARÂMETROS E SCORE INICIAL (Componente IA/ML)
# ====================================================================

# Pontuações Fixas do Modelo de Risco (Usadas na Interface e no Treinamento)
PONTUACAO_RENDA = {'1': 10, '2': 25, '3': 40} # 1: Baixa (1 SM), 2: Média (1-3 SM), 3: Alta (3+ SM)
FATOR_EMPREGO = {'1': 1.0, '2': 2.0}          # 1: Informal, 2: Formal (Fator multiplicativo)
PONTUACAO_HISTORICO = {'1': 0, '2': 50, '3': -20} # 1: Novo, 2: Bom Pagador, 3: Mau Pagador (quitado)
PONTUACAO_TEMPO = {'1': 0, '2': 10, '3': 20} # 1: Novo, 2: 1-5 anos, 3: 5+ anos

# Limite de Risco Fixo para Clientes Novos (Regra de Negócio)
LIMITE_MAX_NOVO_CLIENTE = 1300.00 

def calcular_score_inicial(renda_cat_key, tipo_emprego_key, historico_key, tempo_loja_key):
    """ Calcula o Score Inicial (PD) usando a lógica ponderada. """
    
    pr = PONTUACAO_RENDA.get(renda_cat_key, 0)
    f  = FATOR_EMPREGO.get(tipo_emprego_key, 1.0)
    prp = pr * f

    ph = PONTUACAO_HISTORICO.get(historico_key, 0)
    pt = PONTUACAO_TEMPO.get(tempo_loja_key, 0)
    pa = ph + pt

    score_final = prp + pa
    return score_final

# ---------------------------------------------------------------------
# Geração de Dados Sintéticos (Carteira de Treinamento do AG)
# ---------------------------------------------------------------------

def gerar_carteira_sintetica(tamanho=1000):
    """ Gera um conjunto de dados sintéticos para treinar o AG. """
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

        score = calcular_score_inicial(renda_cat, tipo_emprego, historico, tempo_loja)
        
        valor_compra = random.randint(100, 3000)
        juros = 0.10 
        risco_perda_percentual = max(0.01, min(0.9, (120 - score) / 100)) 
        
        dados.append({
            'score': score,
            'valor_compra': valor_compra,
            'juros_esperado': valor_compra * juros,
            'risco_perda_percentual': risco_perda_percentual
        })
    return dados

# ====================================================================
# PARTE B: ALGORITMO GENÉTICO (AG) - Otimização do Coeficiente C
# ====================================================================

# PARÂMETROS DO AG
TAMANHO_POPULACAO = 100
NUM_GERACOES = 100
TAXA_MUTACAO = 0.05
TAXA_CROSSOVER = 0.8
LIMITE_C_MIN = 0.001 
LIMITE_C_MAX = 2.0    

# Geração da carteira usada no treinamento (executada uma vez)
DADOS_CARTEIRA = gerar_carteira_sintetica(tamanho=1000)

def funcao_aptidao(coeficiente_c):
    """ 
    Função Fitness (Aptidão): MAXIMIZAÇÃO DO LUCRO LÍQUIDO ESPERADO.
    Incentiva a agressividade (bônus de 30%).
    """
    lucro_total = 0
    FATOR_AGRESSIVIDADE = 0.30 
    
    for cliente in DADOS_CARTEIRA:
        score = cliente['score']
        valor_compra = cliente['valor_compra']
        
        if (score / valor_compra) >= coeficiente_c:
            chance_pagar = 1 - cliente['risco_perda_percentual']
            
            ganho_esperado = cliente['juros_esperado'] * chance_pagar
            perda_esperada = valor_compra * cliente['risco_perda_percentual']
            
            bonus_agressividade = valor_compra * FATOR_AGRESSIVIDADE
            
            lucro_na_operacao = ganho_esperado - perda_esperada + bonus_agressividade
            lucro_total += lucro_na_operacao
            
    return lucro_total

def criar_individuo():
    return random.uniform(LIMITE_C_MIN, LIMITE_C_MAX)

def selecao(populacao, aptidoes, k):
    selecionados = []
    candidatos = list(zip(aptidoes, populacao)) 
    for _ in range(k):
        # Seleção por Torneio (3 indivíduos)
        torneio = random.sample(candidatos, 3) 
        melhor = max(torneio, key=lambda x: x[0]) 
        selecionados.append(melhor[1])
    return selecionados

def crossover(pai1, pai2):
    if random.random() < TAXA_CROSSOVER:
        return (pai1 + pai2) / 2
    return random.choice([pai1, pai2])

def mutacao(individuo):
    if random.random() < TAXA_MUTACAO:
        desvio = random.uniform(-0.1, 0.1) 
        individuo += desvio
        return max(LIMITE_C_MIN, min(LIMITE_C_MAX, individuo))
    return individuo

def rodar_ag():
    """ Função principal que executa o AG e retorna o melhor coeficiente C. """
    populacao = [criar_individuo() for _ in range(TAMANHO_POPULACAO)]
    melhor_c_global = None
    melhor_aptidao_global = -np.inf

    for _ in range(NUM_GERACOES):
        aptidoes = [funcao_aptidao(c) for c in populacao]
        melhor_aptidao_atual = max(aptidoes)
        melhor_c_atual = populacao[aptidoes.index(melhor_aptidao_atual)]

        if melhor_aptidao_atual > melhor_aptidao_global:
            melhor_aptidao_global = melhor_aptidao_atual
            melhor_c_global = melhor_c_atual

        nova_populacao = [melhor_c_global] 
        pais = selecao(populacao, aptidoes, TAMANHO_POPULACAO // 2)

        while len(nova_populacao) < TAMANHO_POPULACAO:
            pai1, pai2 = random.choices(pais, k=2)
            filho = crossover(pai1, pai2)
            filho_mutado = mutacao(filho)
            nova_populacao.append(filho_mutado)

        populacao = nova_populacao

    return melhor_c_global