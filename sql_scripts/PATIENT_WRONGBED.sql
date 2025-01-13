WITH CTE1 AS 
(
SELECT	 P.PATIENTNR						AS [Patiëntnummer]
		,P.GESLACHT							AS [Geslacht],
		P.WOONPLAATS						AS [Woonplaats]
		--,P.ADRES							AS [Adres]
		,LEFT(P.POSTCODE,4)					AS [Postcode]
		--,P.HUISNR							AS [Huisnummer]
		--,P.GEBPLAATS						AS [Geboorteplaats]
		--,P.HUISARTS							AS [Huisarts]
		--,P.APOTHEEK							AS [Apotheek]
		,CONVERT(DATE, P.GEBDAT, 102)		AS [Geboortedatum]
		,CONVERT(DATE, P.OVERLDAT,102)		AS [Overlijdensdatum]
		 
FROM HIX_PRODUCTIE.dbo.PATIENT_PATIENT P
)


,
CTE2 AS
(
SELECT	 C1.*
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
LEFT JOIN HIX_PRODUCTIE..OPNAME_OPNAME O	ON C1.Patiëntnummer = O.PATIENTNR
							AND O.OPNDAT BETWEEN '20130101' AND '20240101'
LEFT JOIN HIX_PRODUCTIE..OPNAME_BESTEM B	ON B.CODE = O.BESTEMMING
LEFT JOIN HIX_PRODUCTIE..OPNAME_HERKOMST H	ON H.CODE = O.HERKOMST
WHERE O.WORKFLOWSTATUS IN ('DT000097'/*Opgenomen*/, 'DT000099'/*Ontslagen*/)
AND O.OPNTYPE IN ('0', '1', '3', '4', 'A', 'E', 'F', 'I', 'J', 'K', 'M', 'N', 'O', 'R', 'S', 'T', 'U', 'V', 'W', 'X','Y' ,'Z') -- Als je alleen klinische opnames wilt meenemen, moet dit erbij
AND O.SPECIALISM in ('CHI', 'HLK', 'ORT', 'CHIR')
)

,
CTE3 AS 
(
SELECT DISTINCT  VERH.Referentienummer		AS [Referentienummer]
				,VERH.Patientnummer			AS [Patiëntnummer DBC]
				,VC.DeclCode				AS [Verrichtingcode]
				,VC.DeclOms					AS [Verrichting omschrijving]
				,DBC.DBCNummer				AS [DBCnummer]
				,CONVERT(DATE, DBC.Begindatum, 102)				AS [Begindatum DBC]
				,CONVERT(DATE, DBC.Einddatum, 102)				AS [Einddatum DBC]
				,DBC.Diagnosecode			AS [DBC code]
				,DBC.Specialismecode		AS [DBC Specialisme] 
				,(
					SELECT TOP (1) OMS.diagnoseomschrijving	
					FROM CSDW_Productie..DBCDimDBCCodering OMS
					WHERE OMS.diagnosecodelandelijk = DBC.Diagnosecode
					AND OMS.specialismecode = DBC.Specialismecode
				 )							AS [Diagnose]
FROM CSDW_Productie..VerHftVerrichtingen VERH			
LEFT JOIN CSDW_Productie..VerDftVerrichtingen VERD			ON VERD.VerrichtingSecId = VERH.VerrichtingSecID
LEFT JOIN CSDW_Productie..VerDimVerrichtingcode VC			ON VC.VerrichtingCode_key = VERD.Verrichtingcode_key
LEFT JOIN CSDW_Productie..DBCHftDBCs DBC					ON DBC.DBCNummer = CASE WHEN isnull(VERH.DBCNummer,'') = ''  THEN VERD.DBCNummer ELSE VERH.DBCNummer END
WHERE VC.InvCode IN ( '190218', '190031', '190038')
AND [Status] != 'X'
)

--SELECT	 DISTINCT CTE2.Patiëntnummer as [Patiëntnummer]
SELECT	 DISTINCT 
--HASHBYTES('SHA2_256',CTE2.Patiëntnummer) 
CTE2.Patiëntnummer
as [Patiëntnummer]


		,CTE2.Opnamenummer
		,CTE2.Geslacht
		--,CTE2.Geboortedatum
		--,CTE2.Overlijdensdatum
		,CTE2.[Opnameleeftijd]
		,CTE2.[Overleden tijdens opname]
		,CTE2.[Opname DatumTijd]
		,CTE2.[Ontslag DatumTijd]
		--,CTE2.[Opname jaar]
		,CTE2.[Klinische beddagen (d)]
		,CTE2.[Verkeerd bed ligduur (d)]
		,CASE
				WHEN CTE2.[Klinische beddagen (d)] is null AND CTE2.[Verkeerd bed ligduur (d)] is null THEN 'geen verkeerd bed'
					ELSE 'verkeerd bed'
		END AS [Verkeerd bed]
		,CTE2.Herkomst
		,CTE2.Spoed
		,CTE2.[Totale ligduur (d)]
		,CTE2.[Opname specialisme]
		,CTE2.Ontslagbestemming
		--,CTE3.[Begindatum DBC]
		--,CTE3.[Einddatum DBC]
		,CTE3.[DBC code]
		, CTE2.[Woonplaats]
		--,CTE2.[Adres]
		,LEFT(CTE2.[Postcode], 4) as [Postcode cijfers]
		--,CTE2.[Huisnummer]
		--,CTE2.[Geboorteplaats]
		--,CTE2.[Huisarts]
		--,CTE2.[Apotheek]

		,CTE3.Diagnose

FROM CTE2
LEFT JOIN CTE3	ON CTE2.[Opnamenummer] = CTE3.Referentienummer COLLATE SQL_Latin1_General_CP1_CI_AS
				
			