SELECT 
		--HASHBYTES('SHA2_256', [HIX_PRODUCTIE61].[dbo].[AGENDA_AFSPRAAK].[PATIENTNR]) as PATIENTNR
		[HIX_PRODUCTIE61].[dbo].[AGENDA_AFSPRAAK].[PATIENTNR] as PATIENTNR
		,a.[SPECIALISM]
		,[HIX_PRODUCTIE61].[dbo].[AGENDA_AFSPRAAK].DATUM
		--,[HIX_PRODUCTIE61].[dbo].[CSZISLIB_SPEC].[OMSCHR]
		--,a.[ARTSCODE]
		--,a.[ZOEKCODE]
  FROM [HIX_PRODUCTIE61].[dbo].[CSZISLIB_ARTS] as a
  INNER JOIN [HIX_PRODUCTIE61].[dbo].[AGENDA_AFSPRAAK]  ON [HIX_PRODUCTIE61].[dbo].[AGENDA_AFSPRAAK].[UITVOERDER] = a.[ARTSCODE]
  INNER JOIN [HIX_PRODUCTIE61].[dbo].[CSZISLIB_SPEC]	ON [HIX_PRODUCTIE61].[dbo].[CSZISLIB_SPEC].[SPECCODE] = a.SPECIALISM
  -- An assumption is made here that we only need to take into account hospital visits in the past x years, partly to reduce computation time, but also partly because it sounds reasonable
  WHERE [HIX_PRODUCTIE61].[dbo].[AGENDA_AFSPRAAK].DATUM > '20080101'
