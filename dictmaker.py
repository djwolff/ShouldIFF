f = 'teamid	totalwards	hasAhri	hasAkali	hasAlistar	hasAmumu	hasAnivia	hasAnnie	hasAshe	hasAurelionSol	hasAzir	hasBard	hasBlitzcrank	hasBrand	hasBraum	hasCaitlyn	hasCamille	hasCassiopeia	hasChoGath	hasCorki	hasDarius	hasDiana	hasDraven	hasDrMundo	hasEkko	hasElise	hasEvelynn	hasEzreal	hasFiddlesticks	hasFiora	hasFizz	hasGalio	hasGangplank	hasGaren	hasGnar	hasGragas	hasGraves	hasHecarim	hasHeimerdinger	hasIllaoi	hasIrelia	hasIvern	hasJanna	hasJarvanIV	hasJax	hasJayce	hasJhin	hasJinx	hasKalista	hasKarma	hasKarthus	hasKassadin	hasKatarina	hasKayle	hasKennen	hasKhaZix	hasKindred	hasKled	hasKogMaw	hasLeBlanc	hasLeeSin	hasLeona	hasLissandra	hasLucian	hasLulu	hasLux	hasMalphite	hasMalzahar	hasMaokai	hasMasterYi	hasMissFortune	hasMordekaiser	hasMorgana	hasNami	hasNasus	hasNautilus	hasNidalee	hasNocturne	hasNunu	hasOlaf	hasOrianna	hasPantheon	hasPoppy	hasQuinn	hasRakan	hasRammus	hasRekSai	hasRenekton	hasRengar	hasRiven	hasRumble	hasRyze	hasSejuani	hasShaco	hasShen	hasShyvana	hasSinged	hasSion	hasSivir	hasSkarner	hasSona	hasSoraka	hasSwain	hasSyndra	hasTahmKench	hasTaliyah	hasTalon	hasTaric	hasTeemo	hasThresh	hasTristana	hasTrundle	hasTryndamere	hasTwistedFate	hasTwitch	hasUdyr	hasUrgot	hasVarus	hasVayne	hasVeigar	hasVelKoz	hasVi	hasViktor	hasVladimir	hasVolibear	hasWarwick	hasWukong	hasXayah	hasXerath	hasXinZhao	hasYasuo	hasYorick	hasZac	hasZed	hasZiggs	hasZilean	hasZyra	first_blood_team	firsttower	firstinhib	firstbaron	firstdragon	firstharry'
count=0

print("rowdict = {")
for word in f.split():
    print ('"'+str(count)+'":"' + word + '",')
    count += 1

print("}")