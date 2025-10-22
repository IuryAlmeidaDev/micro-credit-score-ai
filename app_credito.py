import sys
from modelo_avancado import (
    treinar_sistema_completo,
    calcular_score_inicial,
    PONTUACAO_RENDA,
    FATOR_EMPREGO,
    PONTUACAO_HISTORICO,
    PONTUACAO_TEMPO,
    LIMITE_MAX_NOVO_CLIENTE
)

# Mapeamentos para o usuário
MAPA_RENDA = {'1': 'Baixa (até 1 SM)', '2': 'Média (1-3 SM)', '3': 'Alta (3+ SM)'}
MAPA_EMPREGO = {'1': 'Informal', '2': 'Formal'}
MAPA_HISTORICO = {'1': 'Novo Cliente', '2': 'Bom Pagador', '3': 'Mau Pagador (quitado)'}
MAPA_TEMPO = {'1': 'Novo (até 1 ano)', '2': '1-5 anos', '3': '5+ anos'}

def get_input(prompt, options_map, options_dict):
    """Função auxiliar para coletar e validar input numérico"""
    print(prompt)
    for k, v in options_map.items():
        print(f"  [{k}] - {v}")
    key = input("Opção: ").strip()
    if key not in options_dict:
        print("❌ Erro: Opção inválida.")
        return None
    return key

def coletar_input():
    """Coleta as informações do cliente via input"""
    print("\n" + "=" * 60)
    print("  📋 CADASTRO DO CLIENTE")
    print("=" * 60)
    
    # 1. Histórico de Pagamento
    historico_key = get_input("\n1️⃣ Histórico de Pagamento Interno:", 
                              MAPA_HISTORICO, PONTUACAO_HISTORICO)
    if historico_key is None: return None

    # 2. Tempo de Loja (Condicional)
    if historico_key == '1':
        tempo_loja_key = '1'
        print(f"\n2️⃣ Tempo de Loja: [1] - Novo (Assumido automaticamente)")
    else:
        tempo_loja_key = get_input("\n2️⃣ Tempo de Cliente na Loja:", 
                                   MAPA_TEMPO, PONTUACAO_TEMPO)
        if tempo_loja_key is None: return None
    
    # 3. Renda Estimada
    renda_cat_key = get_input("\n3️⃣ Renda Estimada (Capacidade de Pagamento):", 
                             MAPA_RENDA, PONTUACAO_RENDA)
    if renda_cat_key is None: return None

    # 4. Tipo de Emprego
    tipo_emprego_key = get_input("\n4️⃣ Tipo de Emprego (Estabilidade da Renda):", 
                                MAPA_EMPREGO, FATOR_EMPREGO)
    if tipo_emprego_key is None: return None
        
    # 5. Valor da Compra
    try:
        print("\n5️⃣ Valor da Compra (R$): ", end="")
        valor_compra = float(input().replace(',', '.'))
        if valor_compra <= 0: 
            raise ValueError
    except ValueError:
        print("❌ Erro: Valor da compra deve ser um número positivo.")
        return None

    return renda_cat_key, tipo_emprego_key, historico_key, tempo_loja_key, valor_compra

def calcular_metricas_avancadas(score, valor_compra, historico_key, renda_cat_key, 
                                tipo_emprego_key):
    """Calcula métricas avançadas de risco e recomendações"""
    # Estabilidade do perfil
    estabilidade = int(renda_cat_key) * int(tipo_emprego_key)
    
    # Risco estimado
    risco_base = max(0.01, min(0.9, (120 - score) / 100))
    
    if renda_cat_key == '3' and tipo_emprego_key == '2':
        risco_ajustado = risco_base * 0.7
        perfil_risco = "BAIXO"
        cor_risco = "🟢"
    elif historico_key == '3':
        risco_ajustado = min(0.9, risco_base * 1.3)
        perfil_risco = "ALTO"
        cor_risco = "🔴"
    else:
        risco_ajustado = risco_base
        perfil_risco = "MÉDIO"
        cor_risco = "🟡"
    
    # Taxa de juros sugerida
    juros_base = 0.10
    if historico_key == '2' and tipo_emprego_key == '2':
        taxa_juros = juros_base * 0.8
        desc_juros = "com 20% de desconto (bom histórico)"
    elif historico_key == '3':
        taxa_juros = juros_base * 1.5
        desc_juros = "com 50% de acréscimo (risco elevado)"
    else:
        taxa_juros = juros_base
        desc_juros = "padrão"
    
    return {
        'risco_percentual': risco_ajustado * 100,
        'perfil_risco': perfil_risco,
        'cor_risco': cor_risco,
        'taxa_juros': taxa_juros * 100,
        'desc_juros': desc_juros,
        'valor_juros': valor_compra * taxa_juros,
        'valor_total': valor_compra * (1 + taxa_juros),
        'estabilidade': estabilidade
    }

def tomar_decisao_avancada(melhor_c, modelo_rna):
    """Sistema de decisão com análise avançada"""
    print("\n" + "=" * 60)
    print("  🎯 ANÁLISE DE CRÉDITO - SISTEMA AVANÇADO")
    print("=" * 60)
    print(f"\n📌 Política Otimizada: C = {melhor_c:.4f}")
    print("   (Resultado da otimização AG + PSO)")

    cliente_data = coletar_input()
    if cliente_data is None: 
        return

    renda_cat_key, tipo_emprego_key, historico_key, tempo_loja_key, valor_compra = cliente_data

    # SCORE TRADICIONAL
    score_tradicional = calcular_score_inicial(renda_cat_key, tipo_emprego_key, 
                                               historico_key, tempo_loja_key)
    
    # SCORE DINÂMICO (REDE NEURAL)
    features = [
        int(renda_cat_key),
        int(tipo_emprego_key),
        int(historico_key),
        int(tempo_loja_key),
        int(renda_cat_key) * int(tipo_emprego_key),  # estabilidade
        int(tempo_loja_key) * (1 if int(historico_key) == 2 else 0.5),  # experiência
        score_tradicional / valor_compra  # rácio inicial
    ]
    
    score_dinamico = modelo_rna.prever_score(features)
    
    # USA O SCORE DINÂMICO (mais preciso)
    score_final = score_dinamico
    
    # CÁLCULO DO LIMITE
    limite_calculado = score_final / melhor_c
    limite_aplicado = limite_calculado
    
    # Regra de Teto para Clientes Novos
    is_novo_cliente = (historico_key == '1')
    if is_novo_cliente and limite_calculado > LIMITE_MAX_NOVO_CLIENTE:
        limite_aplicado = LIMITE_MAX_NOVO_CLIENTE
    
    # DECISÃO FINAL
    decisao = "APROVAR" if valor_compra <= limite_aplicado else "NEGAR"
    racio = score_final / valor_compra
    
    # MÉTRICAS AVANÇADAS
    metricas = calcular_metricas_avancadas(score_final, valor_compra, historico_key,
                                          renda_cat_key, tipo_emprego_key)
    
    # ========== EXIBIÇÃO DOS RESULTADOS ==========
    print("\n" + "=" * 60)
    print("  📊 ANÁLISE DE CRÉDITO COMPLETA")
    print("=" * 60)
    
    print("\n🔍 SCORES CALCULADOS:")
    print(f"   • Score Tradicional (Regras): {score_tradicional:.2f}")
    print(f"   • Score Dinâmico (RNA):       {score_dinamico:.2f} ⭐")
    print(f"   • Score Final Usado:          {score_final:.2f}")
    
    print(f"\n💰 ANÁLISE FINANCEIRA:")
    print(f"   • Valor Solicitado:     R$ {valor_compra:,.2f}")
    print(f"   • Limite Calculado:     R$ {limite_calculado:,.2f}")
    if is_novo_cliente and limite_calculado > LIMITE_MAX_NOVO_CLIENTE:
        print(f"   • Limite Aplicado:      R$ {limite_aplicado:,.2f} (teto cliente novo)")
    else:
        print(f"   • Limite Aplicado:      R$ {limite_aplicado:,.2f}")
    print(f"   • Rácio Score/Valor:    {racio:.4f}")
    
    print(f"\n⚠️  PERFIL DE RISCO:")
    print(f"   • Classificação:        {metricas['cor_risco']} {metricas['perfil_risco']}")
    print(f"   • Prob. Inadimplência:  {metricas['risco_percentual']:.1f}%")
    print(f"   • Índice Estabilidade:  {metricas['estabilidade']}/6")
    
    print(f"\n💳 CONDIÇÕES SUGERIDAS:")
    print(f"   • Taxa de Juros:        {metricas['taxa_juros']:.1f}% ({metricas['desc_juros']})")
    print(f"   • Valor dos Juros:      R$ {metricas['valor_juros']:.2f}")
    print(f"   • Valor Total:          R$ {metricas['valor_total']:.2f}")
    
    # DECISÃO FINAL COM DESTAQUE
    print("\n" + "=" * 60)
    if decisao == "APROVAR":
        print("  ✅ DECISÃO FINAL: CRÉDITO APROVADO")
        print("=" * 60)
        print(f"\n  ➤ Valor Aprovado: R$ {valor_compra:,.2f}")
        print(f"  ➤ Margem Disponível: R$ {limite_aplicado - valor_compra:,.2f}")
        
        # Recomendações adicionais
        if metricas['perfil_risco'] == "ALTO":
            print("\n  ⚠️  RECOMENDAÇÕES:")
            print("     - Solicitar garantias adicionais")
            print("     - Reduzir prazo de pagamento")
            print("     - Monitoramento intensivo")
        elif metricas['perfil_risco'] == "BAIXO":
            print("\n  🌟 CLIENTE PREMIUM:")
            print("     - Elegível para condições especiais")
            print("     - Considerar aumento de limite futuro")
    else:
        print("  ❌ DECISÃO FINAL: CRÉDITO NEGADO")
        print("=" * 60)
        print(f"\n  ➤ Valor Solicitado: R$ {valor_compra:,.2f}")
        print(f"  ➤ Limite Máximo:    R$ {limite_aplicado:,.2f}")
        print(f"  ➤ Diferença:        R$ {valor_compra - limite_aplicado:,.2f}")
        
        print("\n  💡 ALTERNATIVAS:")
        print(f"     - Reduzir compra para até R$ {limite_aplicado:,.2f}")
        if valor_compra > limite_aplicado * 1.5:
            print("     - Dividir compra em múltiplas transações")
            print("     - Oferecer entrada de 30-40%")
        
        if is_novo_cliente:
            print("\n  ℹ️  INFORMAÇÃO:")
            print(f"     Cliente novo tem limite máximo de R$ {LIMITE_MAX_NOVO_CLIENTE:,.2f}")
            print("     Limite pode aumentar após histórico positivo")
    
    print("\n" + "=" * 60)
    
    # Opção de análise em lote
    print("\n🔄 Deseja analisar outro cliente? (s/n): ", end="")
    continuar = input().strip().lower()
    if continuar == 's':
        tomar_decisao_avancada(melhor_c, modelo_rna)

def menu_principal():
    """Menu interativo do sistema"""
    print("\n" + "=" * 60)
    print("  🏦 SISTEMA AVANÇADO DE ANÁLISE DE CRÉDITO")
    print("     Powered by: Rede Neural + AG + PSO")
    print("=" * 60)
    
    print("\n📋 MENU PRINCIPAL:")
    print("  [1] - Análise Individual de Cliente")
    print("  [2] - Análise em Lote (múltiplos clientes)")
    print("  [3] - Exibir Parâmetros do Sistema")
    print("  [4] - Retreinar Modelo")
    print("  [5] - Teste de Estabilidade (múltiplos treinos)")
    print("  [0] - Sair")
    
    opcao = input("\nEscolha uma opção: ").strip()
    return opcao

def analise_em_lote(melhor_c, modelo_rna):
    """Permite análise de múltiplos clientes sequencialmente"""
    print("\n" + "=" * 60)
    print("  📊 MODO ANÁLISE EM LOTE")
    print("=" * 60)
    
    num_clientes = int(input("\nQuantos clientes deseja analisar? "))
    resultados = []
    
    for i in range(num_clientes):
        print(f"\n{'─' * 60}")
        print(f"  CLIENTE {i+1}/{num_clientes}")
        print(f"{'─' * 60}")
        
        cliente_data = coletar_input()
        if cliente_data is None:
            continue
        
        renda_cat_key, tipo_emprego_key, historico_key, tempo_loja_key, valor_compra = cliente_data
        score_final = calcular_score_inicial(renda_cat_key, tipo_emprego_key, 
                                             historico_key, tempo_loja_key)
        limite_aplicado = score_final / melhor_c
        
        is_novo_cliente = (historico_key == '1')
        if is_novo_cliente and limite_aplicado > LIMITE_MAX_NOVO_CLIENTE:
            limite_aplicado = LIMITE_MAX_NOVO_CLIENTE
        
        decisao = "APROVAR" if valor_compra <= limite_aplicado else "NEGAR"
        
        resultados.append({
            'num': i+1,
            'valor': valor_compra,
            'limite': limite_aplicado,
            'decisao': decisao,
            'score': score_final
        })
        
        print(f"\n✓ Cliente {i+1} analisado: {decisao}")
    
    # Resumo
    print("\n" + "=" * 60)
    print("  📈 RESUMO DA ANÁLISE EM LOTE")
    print("=" * 60)
    
    aprovados = sum(1 for r in resultados if r['decisao'] == 'APROVAR')
    negados = len(resultados) - aprovados
    valor_total_aprovado = sum(r['valor'] for r in resultados if r['decisao'] == 'APROVAR')
    
    print(f"\n  Total Analisado:    {len(resultados)} clientes")
    print(f"  ✅ Aprovados:       {aprovados} ({aprovados/len(resultados)*100:.1f}%)")
    print(f"  ❌ Negados:         {negados} ({negados/len(resultados)*100:.1f}%)")
    print(f"  💰 Volume Aprovado: R$ {valor_total_aprovado:,.2f}")
    
    print("\n  DETALHAMENTO:")
    for r in resultados:
        simbolo = "✅" if r['decisao'] == 'APROVAR' else "❌"
        print(f"    {simbolo} Cliente {r['num']}: R$ {r['valor']:,.2f} | "
              f"Score: {r['score']:.2f} | Limite: R$ {r['limite']:,.2f}")

def exibir_parametros(melhor_c):
    """Mostra os parâmetros do sistema"""
    print("\n" + "=" * 60)
    print("  ⚙️  PARÂMETROS DO SISTEMA")
    print("=" * 60)
    
    print("\n🎯 POLÍTICA DE CRÉDITO:")
    print(f"   Coeficiente C Ótimo: {melhor_c:.4f}")
    print(f"   Limite Novo Cliente: R$ {LIMITE_MAX_NOVO_CLIENTE:,.2f}")
    
    print("\n📊 PONTUAÇÕES DE RENDA:")
    for k, v in MAPA_RENDA.items():
        print(f"   [{k}] {v:25} → {PONTUACAO_RENDA[k]} pontos")
    
    print("\n💼 FATOR EMPREGO:")
    for k, v in MAPA_EMPREGO.items():
        print(f"   [{k}] {v:25} → {FATOR_EMPREGO[k]}x")
    
    print("\n📜 HISTÓRICO:")
    for k, v in MAPA_HISTORICO.items():
        print(f"   [{k}] {v:25} → {PONTUACAO_HISTORICO[k]:+d} pontos")
    
    print("\n⏰ TEMPO DE RELACIONAMENTO:")
    for k, v in MAPA_TEMPO.items():
        print(f"   [{k}] {v:25} → {PONTUACAO_TEMPO[k]} pontos")
    
    print("\n🤖 ARQUITETURA DO MODELO:")
    print("   Rede Neural:  3 camadas (64→32→16 neurônios)")
    print("   Otimização:   Algoritmo Genético + PSO")
    print("   População AG: 100 indivíduos")
    print("   Gerações AG:  100 iterações")
    print("   Partículas PSO: 30 indivíduos")
    print("   Dados de treino: 2000 clientes sintéticos (fixos)")

def teste_estabilidade():
    """Testa a estabilidade do treinamento mostrando que o resultado é sempre o mesmo"""
    from modelo_avancado import otimizacao_hibrida, SEED_GLOBAL
    import numpy as np
    
    print("\n" + "=" * 60)
    print("  🔬 TESTE DE ESTABILIDADE DO ALGORITMO")
    print("=" * 60)
    
    print("\n⚠️  IMPORTANTE: Como a carteira é FIXA e usa seed determinística,")
    print("   o resultado deve ser IDÊNTICO em todas as execuções!")
    
    num_testes = int(input("\nQuantos treinos deseja executar? (recomendado: 3-5): "))
    
    resultados = []
    
    print("\n" + "─" * 60)
    for i in range(num_testes):
        print(f"\n🔄 Treinamento {i+1}/{num_testes}...")
        melhor_c = otimizacao_hibrida(verbose=False)
        resultados.append(melhor_c)
        print(f"   Resultado: C = {melhor_c:.6f}")
    
    print("\n" + "=" * 60)
    print("  📊 ANÁLISE DE ESTABILIDADE")
    print("=" * 60)
    
    print(f"\n🎯 Resultados de {num_testes} treinamentos:")
    for i, c in enumerate(resultados, 1):
        print(f"   Treino {i}: C = {c:.6f}")
    
    media = np.mean(resultados)
    desvio = np.std(resultados)
    minimo = np.min(resultados)
    maximo = np.max(resultados)
    variacao = ((maximo - minimo) / media * 100) if media > 0 else 0
    
    print(f"\n📈 Estatísticas:")
    print(f"   Média:            {media:.6f}")
    print(f"   Desvio Padrão:    {desvio:.6f}")
    print(f"   Mínimo:           {minimo:.6f}")
    print(f"   Máximo:           {maximo:.6f}")
    print(f"   Variação:         {variacao:.4f}%")
    
    if desvio < 0.0001:
        print("\n✅ EXCELENTE! Variação praticamente ZERO.")
        print("   O algoritmo é completamente estável!")
    elif desvio < 0.001:
        print("\n✅ ÓTIMO! Variação mínima detectada.")
        print("   Isso é normal devido a arredondamentos numéricos.")
    else:
        print("\n⚠️  ATENÇÃO! Há variação significativa.")
        print("   Verifique se a seed está sendo aplicada corretamente.")
    
    print(f"\n💡 Seed Global: {SEED_GLOBAL}")
    print("   Mudando a seed, você obtém diferentes carteiras de treinamento.")

def teste_robustez():
    """Testa robustez com DIFERENTES carteiras (seeds variadas)"""
    from modelo_avancado import otimizacao_hibrida, gerar_carteira_sintetica_avancada
    import numpy as np
    
    print("\n" + "=" * 60)
    print("  🧪 TESTE DE ROBUSTEZ (VALIDAÇÃO CRUZADA)")
    print("=" * 60)
    
    print("\n💡 Este teste treina com DIFERENTES carteiras sintéticas")
    print("   para verificar se o modelo é ROBUSTO ou SENSÍVEL aos dados.")
    
    num_seeds = int(input("\nQuantas carteiras diferentes testar? (recomendado: 5-10): "))
    
    resultados = []
    seeds_usadas = []
    
    print("\n" + "─" * 60)
    for i in range(num_seeds):
        seed = 42 + i * 100  # Seeds: 42, 142, 242, 342...
        seeds_usadas.append(seed)
        
        print(f"\n🌱 Carteira {i+1}/{num_seeds} (seed={seed})...")
        
        # Força regeneração da carteira com nova seed
        import modelo_avancado
        modelo_avancado.SEED_GLOBAL = seed
        modelo_avancado._CARTEIRA_CACHE = None  # Limpa cache
        modelo_avancado.DADOS_CARTEIRA = None
        
        melhor_c = otimizacao_hibrida(verbose=False)
        resultados.append(melhor_c)
        print(f"   Resultado: C = {melhor_c:.6f}")
    
    # Restaura seed original
    import modelo_avancado
    modelo_avancado.SEED_GLOBAL = 42
    modelo_avancado._CARTEIRA_CACHE = None
    modelo_avancado.DADOS_CARTEIRA = None
    
    print("\n" + "=" * 60)
    print("  📊 ANÁLISE DE ROBUSTEZ")
    print("=" * 60)
    
    print(f"\n🎯 Resultados com {num_seeds} carteiras diferentes:")
    for i, (c, seed) in enumerate(zip(resultados, seeds_usadas), 1):
        print(f"   Seed {seed:4d}: C = {c:.6f}")
    
    media = np.mean(resultados)
    desvio = np.std(resultados)
    minimo = np.min(resultados)
    maximo = np.max(resultados)
    variacao = ((maximo - minimo) / media * 100) if media > 0 else 0
    
    print(f"\n📈 Estatísticas:")
    print(f"   Média:            {media:.6f}")
    print(f"   Desvio Padrão:    {desvio:.6f}")
    print(f"   Mínimo:           {minimo:.6f}")
    print(f"   Máximo:           {maximo:.6f}")
    print(f"   Variação:         {variacao:.2f}%")
    print(f"   IC 95%:           [{media - 2*desvio:.6f}, {media + 2*desvio:.6f}]")
    
    # Interpretação para produção
    print("\n" + "─" * 60)
    print("  💼 INTERPRETAÇÃO PARA PRODUÇÃO:")
    print("─" * 60)
    
    if variacao < 5:
        print("\n✅ MODELO MUITO ROBUSTO (variação < 5%)")
        print("   ➤ Seguro para produção")
        print("   ➤ Resultado estável independente dos dados")
    elif variacao < 15:
        print("\n⚠️  MODELO MODERADAMENTE ROBUSTO (variação 5-15%)")
        print("   ➤ Aceitável para produção com monitoramento")
        print("   ➤ Considere usar a MÉDIA dos treinos")
    elif variacao < 30:
        print("\n⚠️  MODELO SENSÍVEL (variação 15-30%)")
        print("   ➤ Use com cautela em produção")
        print("   ➤ Recomenda-se ensemble ou retreinamento frequente")
    else:
        print("\n❌ MODELO INSTÁVEL (variação > 30%)")
        print("   ➤ NÃO recomendado para produção")
        print("   ➤ Revise hiperparâmetros ou arquitetura")
    
    print(f"\n💡 Recomendação: Use C médio = {media:.6f} em produção")
    print(f"   com monitoramento se C real variar ± {desvio*2:.6f}")

# ====================================================================
# EXECUÇÃO PRINCIPAL DO SISTEMA
# ====================================================================

if __name__ == "__main__":
    # Treinamento inicial
    melhor_c_global, modelo_rna_global = treinar_sistema_completo()
    
    # Loop do menu
    while True:
        opcao = menu_principal()
        
        if opcao == '1':
            tomar_decisao_avancada(melhor_c_global, modelo_rna_global)
        
        elif opcao == '2':
            analise_em_lote(melhor_c_global, modelo_rna_global)
        
        elif opcao == '3':
            exibir_parametros(melhor_c_global)
        
        elif opcao == '4':
            print("\n🔄 Retreinando o sistema...")
            melhor_c_global, modelo_rna_global = treinar_sistema_completo()
            print("\n✅ Sistema retreinado com sucesso!")
        
        elif opcao == '5':
            teste_estabilidade()
        
        elif opcao == '0':
            print("\n👋 Encerrando sistema. Até logo!")
            break
        
        else:
            print("\n❌ Opção inválida. Tente novamente.")
        
        input("\n⏎ Pressione ENTER para continuar...")