WITH CTE1 AS 
(
SELECT	 HASHBYTES('SHA2_256',P.PATIENTNR)	AS [Patiëntnummer]
		,P.GESLACHT							AS [Geslacht]
		,P.WOONPLAATS						AS [Woonplaats]
		,P.ADRES							AS [Adres]
		,P.POSTCODE							AS [Postcode]
		--,P.HUISNR							AS [Huisnummer]
		--,P.GEBPLAATS						AS [Geboorteplaats]
		,P.HUISARTS							AS [Huisarts]
		,P.APOTHEEK							AS [Apotheek]
		,CONVERT(DATE, P.GEBDAT, 102)		AS [Geboortedatum]
		,CONVERT(DATE, P.OVERLDAT,102)		AS [Overlijdensdatum]
		 
FROM HIX_PRODUCTIE.dbo.PATIENT_PATIENT P
)

SELECT	 C1.Patiëntnummer
		,O.PLANNR												AS [Opnamenummer]
		,DATEDIFF(YY, C1.Geboortedatum, O.OPNDAT) - 
										CASE	
											WHEN RIGHT(CONVERT(VARCHAR(6), O.OPNDAT, 12), 4) >= RIGHT(CONVERT(VARCHAR(6), C1.Geboortedatum, 12), 4) THEN 0 
											ELSE 1 
										END						AS [Opnameleeftijd]
		,O.OPNDAT + O.OPNTIJD									AS [Opname DatumTijd]
		,YEAR(O.OPNDAT)											AS [Opname jaar]
		,(	
			SELECT TOP (1) M.INGDAT
			FROM [HIX_PRODUCTIE].[dbo].[OPNAME_OPNMUT] M
			WHERE M.PLANNR = O.PLANNR
			AND M.OPNTYPE IN ('4', 'F', 'V') --verkeerd bed codes
		 )														AS [Verkeerd bed startdatum]
		,O.ONTSLDAT + O.ONTSLTIJD								AS [Ontslag DatumTijd]
		,DATEDIFF(DAY, O.OPNDAT, (SELECT TOP (1) M.INGDAT
								  FROM [HIX_PRODUCTIE].[dbo].[OPNAME_OPNMUT] M
								  WHERE M.PLANNR = O.PLANNR
								  AND M.OPNTYPE IN ('4', 'F', 'V')
								  ))							AS [Klinische beddagen (d)]
		,DATEDIFF(DAY, (SELECT TOP (1) M.INGDAT
						FROM [HIX_PRODUCTIE].[dbo].[OPNAME_OPNMUT]  M
						WHERE M.PLANNR = O.PLANNR
						AND M.OPNTYPE IN ('4', 'F', 'V')
						), O.ONTSLDAT)							AS [Verkeerd bed ligduur (d)]
		,DATEDIFF(DAY, O.OPNDAT, O.ONTSLDAT) +1					AS [Totale ligduur (d)]	
		,O.SPECIALISM											AS [Opname specialisme]
		,O.SPOED												AS [Spoed]
		,H.OMSCHR												AS [Herkomst]
		,B.OMSCHR												AS [Ontslagbestemming]
		,CASE
				WHEN B.OMSCHR LIKE '%overleden%' THEN 'Ja'
				ELSE 'Nee'
		 END													AS [Overleden tijdens opname]
		
FROM CTE1 C1
LEFT JOIN HIX_PRODUCTIE..OPNAME_OPNAME O	ON C1.Patiëntnummer = HASHBYTES('SHA2_256', O.PATIENTNR)
							AND O.OPNDAT BETWEEN '20130101' AND '20240101'
LEFT JOIN HIX_PRODUCTIE..OPNAME_BESTEM B	ON B.CODE = O.BESTEMMING
LEFT JOIN HIX_PRODUCTIE..OPNAME_HERKOMST H	ON H.CODE = O.HERKOMST
WHERE O.WORKFLOWSTATUS IN ('DT000097'/*Opgenomen*/, 'DT000099'/*Ontslagen*/)
AND O.OPNTYPE IN ('0', '1', '3', '4', 'A', 'E', 'F', 'I', 'J', 'K', 'M', 'N', 'O', 'R', 'S', 'T', 'U', 'V', 'W', 'X','Y' ,'Z') -- Als je alleen klinische opnames wilt meenemen, moet dit erbij
AND O.SPECIALISM in ('CHI', 'HLK', 'ORT', 'CHIR')