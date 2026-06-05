# Consultoria em Nowcasting para Vigilância Epidemiológica

## Quem é Marcos

Marcos Malveira — atualmente na SESACRE, vencedor do Prêmio Dionísio Herrera (EXPOEPI 2026). Especialista em modelagem epidemiológica e nowcasting, com atuação direta na implantação de sistemas de estimativa em tempo real para arboviroses. Criador do pacote **nowcasting-sus**, ferramenta opensource que já está rodando em produção em estados brasileiros, gerando boletins automáticos semanais sem intervenção manual. O foco é entregar dado útil para tomada de decisão — não relatórios acadêmicos.

---

## O que ofereço

### 1. Implantação de Nowcasting em Estados e Municípios

Configuração completa do pipeline de nowcasting: desde a coleta dos dados de notificação (SINAN, SIVEP-Gripe, sistemas locais), passando pela calibração dos modelos Bayesianos (PyMC, Stan), até a geração automatizada do boletim semanal. O sistema roda sozinho — a equipe de vigilância recebe o PDF/Dashboard pronto sem precisar abrir terminal.

**O que está incluído:**
- Configuração do ambiente com nowcasting-sus
- Calibração do modelo para a realidade local (sazonalidade, atraso de notificação, feriados)
- Geração automática de boletim em PDF/HTML
- Documentação da operação para equipe técnica

### 2. Treinamento In Loco ou Remoto para Equipes de Vigilância

Treinamento prático — não teórico. A equipe sai apta a interpretar as curvas de nowcasting, validar os resultados e identificar alertas precoces. Conteúdo adaptado ao nível técnico da equipe (estatísticos, epidemiologistas, técnicos de informação).

**Tópicos do treinamento:**
- O que é nowcasting e por que funciona para arboviroses e síndromes respiratórias
- Interpretação de gráficos: casos esperados vs. observados, limite epidêmico, tendência de curto prazo
- Operação do sistema: como gerar o boletim, como verificar se os dados estão consistentes
- Noções de modelagem Bayesiana aplicada (sem matemática pesada — só o essencial)

### 3. Desenvolvimento de Dashboards e Boletins Automatizados

Criação de dashboards interativos (Quarto/R Markdown, Streamlit, ou HTML estático) e boletins automatizados em PDF. Integração direta com o pipeline de nowcasting — o dashboard é atualizado automaticamente a cada nova rodada de dados.

**Exemplos de entrega:**
- Boletim semanal em PDF com tabelas, gráficos e interpretação automática
- Dashboard público ou restrito para monitoramento em tempo real
- Relatório automatizado por e-mail ou Telegram

### 4. Consultoria em Modelagem de Dados Epidemiológicos (PyMC, Stan)

Apoio na construção de modelos Bayesianos para problemas específicos: nowcasting, estimativa de subnotificação, projeção de cenários, correção por atraso de notificação. Modelos validados e prontos para produção, com inferência via MCMC (PyMC ou Stan).

**Quando contratar essa consultoria:**
- Você já tem os dados, mas o modelo não converge ou os resultados não são confiáveis
- Precisa adaptar o nowcasting-sus para um agravo diferente (ex.: COVID-19, leptospirose, influenza)
- Quer validar um modelo existente com benchmarks e testes de sensibilidade
- Precisa de um modelo sob medida para um artigo ou relatório técnico

---

## Como Funciona

**1. Diagnóstico** (1–2 semanas)
Entendemos o fluxo de dados, a infraestrutura disponível e a capacidade técnica da equipe. Identificamos gargalos de qualidade dos dados e definimos os indicadores prioritários.

**2. Implementação** (2–4 semanas)
Configuração do nowcasting-sus, calibração do modelo, geração do primeiro boletim/dashboard. Validação com dados históricos e ajuste fino. A equipe acompanha cada etapa.

**3. Acompanhamento** (1–3 meses)
Suporte remoto contínuo para correção de rumo, ajustes no modelo e treinamento complementar. O sistema já está rodando — o acompanhamento garante que continue gerando resultados confiáveis.

---

## Público-Alvo

- **Secretarias Estaduais de Saúde (SES)** — implantação em âmbito estadual, treinamento de equipes regionais
- **Secretarias Municipais de Saúde** — nowcasting para cidades de médio e grande porte
- **OPAS / OMS** — apoio técnico a países da América Latina
- **Fiocruz** — colaboração em projetos de pesquisa aplicada e vigilância
- **ONGs e fundações** — projetos de fortalecimento da vigilância epidemiológica

---

## Contato

**Email:** mvmlima@hotmail.com

Consultoria baseada em evidências e em soluções que já estão rodando em campo. Sem teoria — só o que funciona.
