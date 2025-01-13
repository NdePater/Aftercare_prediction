DROP TABLE #TempStr;
DROP TABLE #TempInt;
DROP TABLE #temp_table;


CREATE TABLE #TempStr (Value VARCHAR(255));
INSERT INTO #TempStr (Value)
VALUES 
--dyspnoe
('CS00059513'), 
('CS00127453'), 
('LA00015770'),
--ASA
('LA00011607'), 
('CS00072268'), 
('CS00056727'), 
('CS00087377'), 
('CS00165707'), 
('CS00165590'), 
('CS00165561'), 
('CS00165777'), 
('CS00165740'), 
('CS00086982'), 
('0000130216'), 
('0000131636'), 
('CS00087322'), 
('CS00087020'),
--BMI
('0000108305'),
('0000127166'),
('0000132459'),
('0000133365'),
('0000133930'),
('0000136215'),
('CS00001071'),
('CS00006278'),
('CS00012904'),
('CS00017191'),
('CS00017194'),
('CS00029078'),
('CS00052926'),
('CS00061798'),
('CS00081398'),
('CS00092782'),
('CS00097063'),
('CS00099714'),
('CS00109954'),
('CS00205722'),
('CS00221898');

CREATE TABLE #TempInt (Value VARCHAR(255));
INSERT INTO #TempInt (Value)
VALUES 
--ASA
('130216'), 
('131636'), 
--BMI
('108305'),
('127166'),
('132459'),
('133365'),
('133930'),
('136215');


--CREATE TABLE #temp_table (
--	REALVRID VARCHAR(255), 
--	PATIENTNR VARCHAR(255), 
--	DATUM VARCHAR(255), 
--	BEANTWID VARCHAR(255), 
--	ANTWOORD VARCHAR(255), 
--	OBJALIAS VARCHAR(255), 
--	OBJID VARCHAR(255));
SELECT CAST(REALVRID AS VARCHAR) as REALVRID, PATIENTNR, DATUM, BEANTWID, ANTWOORD, OBJALIAS, OPSLAGID, OBJID
INTO #temp_table
from dbo.VRLIJST_VROPSLG; 

with EPISODE_DIAG as (
		-- The outer select clause serves as constraint validation
       select SPECIALISM, CODE, DATUM, IIF(VOLGENDE_DATUM <= EINDDATUM, DATEADD(DAY, -1, VOLGENDE_DATUM), EINDDATUM) AS EINDDATUM, OMSCHRIJV
       from (
	   -- This select clause returns the different coding schemes used over the years. It contains the specialism, related code, and the period it was in use.
             select SPECIALISM, CODE
                    , isnull(DATUM, datefromparts(1900,1,1)) as DATUM
                    , isnull(EINDDATUM, datefromparts(2099,12,31)) as EINDDATUM
					-- This line selects the next date, in a partition containing the same specialism, code.
					-- This way, it is known when the next period starts.
                    , LEAD(DATUM) OVER (PARTITION BY SPECIALISM, CODE ORDER BY ISNULL(DATUM, DATEFROMPARTS(1900,1,1))) AS VOLGENDE_DATUM
                    , OMSCHRIJV
             from dbo.EPISODE_DIAG
       ) s 
       where DATUM < VOLGENDE_DATUM or VOLGENDE_DATUM is null

)

SELECT TOP 1 WITH TIES	
--HASHBYTES('SHA2_256', A.PATIENTNR)	
A.PATIENTNR
AS [PatientID] 
		, A.BEANTWID		AS [FormID]
		, A.OPSLAGID		AS [Question AnswerID]
		, A.DATUM			AS [Date Question]
		-- In the ps table there is also OK ID
		, ps.DATUM			AS [ScreenData]
		, ps.OPNAMENR		AS [OpnameNr]
		--, ps.OPERATIE		AS [TypeOperation]
		, A.REALVRID		AS [QuestionID]
		,V.STELLING			AS [Discription Question]
		-- below line was meant to give the correct answer, but did not.
		--,CASE v.ENKELV WHEN 1 then isnull(EA.ANTWOORD, A.ANTWOORD) ELSE A.ANTWOORD END AS [OpenTekst Answer] 
		,ISNULL(Y.OMSCHR, A.ANTWOORD) as [Antwoord]
		--,ff.BEGINDAT		AS [Period Start Date]
		--,ff.EINDDAT			AS [Period End Date]
    	,ff.SPECIALISM		AS [Physician department Name]
-- Take the patient information
FROM [HIX_PRODUCTIE61].[dbo].[PATIENT_PATIENT] as vv
-- combine with EPISODE (hospital visits?) data
INNER JOIN dbo.EPISODE_EPISODE as ss ON vv.[PATIENTNR] = ss.[PATIENTNR]
-- combine with detailed info EPISODE
INNER JOIN [HIX_PRODUCTIE61].[dbo].[EPISODE_DBCPER] as ff ON ff.EPISODE = ss.EPISODE
-- combine with information about specialism (generated above)
INNER JOIN EPISODE_DIAG ED ON ED.CODE = ff.HOOFDDIAG AND 
ED.SPECIALISM = ff.SPECIALISM AND ff.BEGINDAT BETWEEN ED.DATUM AND ED.EINDDATUM
-- combine with all questions asked to the patient, related to that episode
LEFT JOIN #temp_table A ON A.PATIENTNR = ss.PATIENTNR
	AND A.DATUM BETWEEN ff.BEGINDAT AND ISNULL(ff.EINDDAT, DATEFROMPARTS(2099,12,31))
-- combine with detailed information about question
LEFT JOIN dbo.VRLIJST_VRAGEN V ON A.REALVRID = V.VRAAGID 
-- combine with information about questionlist
LEFT JOIN dbo.VRLIJST_LSTOPSLG L ON A.BEANTWID = L.OPSLAGID 
-- in certain question list, codes are used as answers. The meaning of these codes are stored in VRLIJST_LSTOPSLG
LEFT JOIN VRLIJST_KEUZELST Y    ON A.ANTWOORD = Y.CODE
-- combine with more information about the VRLIJST
LEFT OUTER JOIN dbo.VRLIJST_LIJSTDEF D ON L.LIJSTID = D.LIJSTID 
	OUTER APPLY (
	  SELECT top 1 ev.ANTWOORD FROM dbo.VRLIJST_VROPSLG EV
	  -- A line added to prevent double observations
	  WHERE V.ENKELV = Cast(1 as bit) 
	  AND EV.REALVRID = A.REALVRID 
	  AND EV.OBJALIAS = A.OBJALIAS 
	  AND EV.OBJID = A.OBJID 
	  -- Unclear. Multiple ANTWOORDen, where the top one is important?
	  ORDER BY EV.BEANTWID DESC, EV.OPSLAGID DESC) EA
-- combine with information about preoperative screening, related to episode
LEFT JOIN dbo.PREOP_SCREEN ps ON ss.PATIENTNR = ps.PATIENTNR
WHERE ps.DATUM BETWEEN ff.BEGINDAT AND ISNULL(ff.EINDDAT, DATEFROMPARTS(2099,12,31)) 
AND ff.BEGINDAT BETWEEN '20120601' AND '20240101' 
--AND A.DATUM BETWEEN '20230101' AND '20230201'
AND ff.SPECIALISM in ('ORT', 'HLK', 'CHI', 'CHIR')
AND CAST(A.REALVRID AS VARCHAR) in (SELECT Value FROM #TempStr)
ORDER BY ROW_NUMBER() OVER (PARTITION BY A.OPSLAGID ORDER BY ff.BEGINDAT DESC, ff.DBCNUMMER DESC)