/*0.
We are exploring a database from the SQL Data analysis course I did from Udemy. This is a database containing 
insurance information pertaining to member information, their respective covers & claims made for the years 2001 to 2014
First we want to have a look at the member data table to see if any cleaning, manipulation and or transformation
*/
select *
from [dbo].[Member]

/*
From a quick cursory look, we can see that we have fields containing demographic information, occupation & job status information along with 
business enterprise information such date when each member joined the fund

The first thing we woould like to do is get some descriptive stats of the membership base. We have decided to deal with null values on a case by case basis
for each query thus we check is any of the fields we're about to use contain nulls
*/
/*
Null value query for the information we want to work with:

select 
(select count(memberkey) from member where memberkey is null) as total_nulls_members
,(select count(memberkey) from member where gender is null) as total_nulls_gender
,(select count(memberkey) from member where age_band is null) as total_nulls_age
from Member
group by gender

1. We can see that there are no null values in the memberkey, gender or age_band fields
*/
select
	count (memberkey) as TotalMembers_byGenderandAge
	,cast(count (memberkey)*100.0/MemberCount.Total as decimal (10,2))as Pct_of_TotalMembers
	,gender
	,age_band
	,Total

from Member	
	,(select count (memberkey) as Total from member) MemberCount

group by
	gender
	,age_band
	,MemberCount.Total

order by 
	age_band

/*2.
We look at the membership base according to their occupational information
Fisrt things first, we need to check for null values or values that may misrepresent our data
*/
/*
Select
employee_status,
(select count(memberkey) from member where annual_salary is null or annual_salary = ' ') as total_nulls_annual_salary
,(select count(memberkey) from member where occupation is null  or occupation = ' ') as total_nulls_occupation
,(select count(memberkey) from member where employee_status is null or employee_status = ' ') as total_nulls_employee_status

from Member
--where 
--	--annual_salary = 25000
--	occupation = 'Not Stated'
--	or employee_status not in ('1) High Earner','2) Medium Earner','3) Low Earner','4) Inactive','5) New Members')
group by employee_status
order by employee_status

We see there are no nulls but when we take a closer look we see that the employee status contains incorrect information 
and/or empty entries not marked as nulls as well as the occupation having a 'Not stated' entry. Check the notes for how I dealth with them
*/

Select
	cast(avg(mem.annual_salary) as decimal(25,2)) as AverageSalary
	--,annual_salary
	,count(mem.MemberKey) as MemberCount
	,mem.occupation
	,mem.Employee_Status_Updated

from
	(Select *,

	Case
	When employee_status not in ('1) High Earner','2) Medium Earner','3) Low Earner','4) Inactive','5) New Members')
	Then '4) Inactive'
	Else employee_status
	end as Employee_Status_Updated

	from Member) as mem

--where 
--	annual_salary = 25000
--	and occupation = 'Not Stated'
--	--and employee_status not in ('1) High Earner','2) Medium Earner','3) Low Earner','4) Inactive','5) New Members')

group by
	occupation
	,Employee_Status_Updated
	

order by
	occupation
	,Employee_Status_Updated

/* 3.
Now we would like to see the geographical information along our membership details to breakdown membership per state(Australia)
The geographical info is located on a seperate data table which means we need to bring that info onto our member's table
*/

select 
	PostalCode,state

from [dbo].[PostalCode]
where PostalCode != ' '
group by PostalCode,state
order by PostalCode desc,State

/*
First thing to note (as a non Australian) some of the postal codes are repeated over different surburbs which are in the same state. Seeing as we
want to breakdown our membership demographics over the state lines, this won't be an issue so we will keep that in mind
*/
select 
	count(memberkey) as Members_byState
	,state
	,CountryNew
/*
We created views to join with that removed empty entries in the postal code table & member table to avoid the join creating duplicates of null entries
*/
from
	(select
		memberkey, 
		case
		when postal_code != ' ' and country is null then 'Australia'
		else country
		end as CountryNew,
		Case
		when postal_code = ' ' then '0000'
		else postal_code
		end AS Postal_CodeNew
		from member
	) AS mem
inner join 
	(select PostalCode,state from [dbo].[PostalCode]
	where PostalCode != ' ' group by PostalCode ,state
	) AS pos 
	on mem.postal_codenew = pos.PostalCode

group by state,CountryNew
order by CountryNew,state

/* 4.
Next We want to have a look at our member's cover details so we need to ensure there are no nulls in the member cover table;
The below technique was not taught in the course and subsequent reseach stated that dynamic queries would be a more effecient way of counting the nulls
however, since the table was not too large I mannually copied all the column names for the sack of showcasing one method of counting nulls
*/
SELECT 
	count(memberkey) as Total_Members_Covered
	,SUM (case when memberkey is null or memberkey = ' ' then 1 else 0 end ) as TotalNull_CountMem
	,SUM (case when underwriting_year is null or underwriting_year = ' ' then 1 else 0 end ) as TotalNull_CountUnd
	,SUM (case when total_death_cover is null or total_death_cover = ' ' then 1 else 0 end ) as TotalNull_CountDCover
	,SUM (case when total_death_cover_premium is null or total_death_cover_premium = ' ' then 1 else 0 end ) as TotalNull_CountDPrem
	,SUM (case when total_ip_cover is null or total_ip_cover = ' ' then 1 else 0 end ) as TotalNull_CountIP
	,SUM (case when total_ip_cover_premium is null or total_ip_cover_premium = ' ' then 1 else 0 end ) as TotalNull_CountIPPrem
	,SUM (case when total_tpd_cover is null or total_tpd_cover = ' ' then 1 else 0 end ) as TotalNull_CountTPD
	,SUM (case when total_tpd_cover_premium is null or total_tpd_cover_premium = ' ' then 1 else 0 end ) as TotalNull_CountTPDPrem
  FROM [Chapter 4 - Insurance].[dbo].[MemberCover]

/* 
We want to have a look at the cover details of our members per the entreprise (product supplier) to have a view of which enterprise provides the most cover
for all our members and the cover details in terms of the average cover value, premiums and total member count
*/
SELECT 
count(mcover.MemberKey) as Member_Count, underwriting_year, 
AVG(case when total_death_cover is null or total_death_cover = ' ' then 0 else total_death_cover end )AS AVG_Total_Death_Cover,
AVG(case when total_death_cover_premium is null or total_death_cover_premium = ' ' then 0 else total_death_cover_premium end) AS AVG_total_death_cover_premium,
AVG(total_ip_cover) AVG_IP_Cover, 
AVG(total_ip_cover_premium) AVG_IP_Prem,
AVG(total_tpd_cover) AVG_TPD_Cover,
AVG(total_tpd_cover_premium) AVG_TPD_Prem,
mem.EnterpriseKey, ent.EnterpriseName

from MemberCover mcover inner join Member mem on mcover.MemberKey = mem.MemberKey inner join Enterprise Ent on mem.EnterpriseKey = ent.EnterpriseKey
group by underwriting_year,mem.EnterpriseKey,ent.EnterpriseName
order by underwriting_year,member_count desc,AVG_total_death_cover_premium DESC,AVG_IP_Prem DESC,AVG_TPD_Prem DESC

/* 5.
Now we want to have a breakdown of the member cover details to analysis if there are any gaps in our membership for improving/extending their cover
We need an analysis of the cover details per cover type across occupational groups
*/
SELECT 
count(mcover.MemberKey) as Member_Count, underwriting_year,
AVG(mem.annual_salary) AVG_Annual_Salary,
AVG(case when total_death_cover is null or total_death_cover = ' ' then 0 else total_death_cover end )AS AVG_Total_Death_Cover,
AVG(case when total_death_cover_premium is null or total_death_cover_premium = ' ' then 0 else total_death_cover_premium end) AS AVG_total_death_cover_premium,
AVG(total_ip_cover) AVG_IP_Cover, 
AVG(total_ip_cover_premium) AVG_IP_Prem,
AVG(total_tpd_cover) AVG_TPD_Cover,
AVG(total_tpd_cover_premium) AVG_TPD_Prem,
occupation

from MemberCover mcover inner join Member mem on mcover.MemberKey = mem.MemberKey 
group by underwriting_year,occupation
order by underwriting_year,member_count desc,AVG_total_death_cover_premium DESC,AVG_IP_Prem DESC,AVG_TPD_Prem DESC

/* 6.
We want to have a look at our claims history as this will inform how we design our products in future. First we want to breakdown the claim events
per gender and what the average age of the member was for the claim.

*/
select
count(cl.memberkey) Total_Claims
,sum(claimpaidamount) as Total_ClaimPaid
,gender
,AVG(age) AVG_Member_Age
,ClaimType

from MemberClaims cl inner join Member mem on cl.MemberKey = mem.MemberKey
where claimstatus = 'Paid'
group by ClaimType,gender
order by ClaimType

/*

*/
select
count(cl.memberkey) Total_Claims
,sum(claimpaidamount) as Total_ClaimPaid
,gender
,AVG(age) AVG_Member_Age
,ClaimType
,YEAR(claimcreateddate) Claim_Year

from MemberClaims cl inner join Member mem on cl.MemberKey = mem.MemberKey
where claimstatus = 'Paid'
group by ClaimType,YEAR(claimcreateddate),gender
order by YEAR(claimcreateddate),ClaimType

/* 7.
We want to have a look at the profitability across all our cover range to see which years were profitable for the business
*/
Create view YearlyPremium AS
	select
		underwriting_year 
		,sum([total_death_cover_premium]) as DTHCoverPremium
		,sum([total_tpd_cover_premium])	  as TPDCoverPremium
		,sum([total_ip_cover_premium])    as IPCoverPremium
	from [dbo].[MemberCover] mc
	group by underwriting_year;
/*
We created a view as we want to call it in subsequent queries. To avoid repeatedly calling the same predicates in correlated subqueries we then 
can have a view from which we can call from
*/
select 
	 YearlyPremium.underwriting_year
	,cl.ClaimType
	,YearlyPremium.TPDCoverPremium
	,sum(claimpaidamount) as TotalClaimPaid
	,YearlyPremium.TPDCoverPremium - sum(cl.claimpaidamount) as CoverProfit 
from
	[dbo].[MemberClaims] cl

outer apply YearlyPremium

where
	underwriting_year is not null and ClaimType ='TPD'
	and year(claimpaiddate) between 2010 and 2014
group by 
	 YearlyPremium.underwriting_year
	,YearlyPremium.TPDCoverPremium
	,cl.ClaimType 
UNION
select 
	 YearlyPremium.underwriting_year
	,cl.ClaimType
	,YearlyPremium.DTHCoverPremium
	,sum(claimpaidamount) as TotalClaimPaid
	,YearlyPremium.DTHCoverPremium - sum(cl.claimpaidamount) as CoverProfit 
from
	[dbo].[MemberClaims] cl

outer apply YearlyPremium

where
	underwriting_year is not null and ClaimType ='DTH'
	and year(claimpaiddate) between 2010 and 2014
group by 
	 YearlyPremium.underwriting_year
	,YearlyPremium.DTHCoverPremium
	,cl.ClaimType 
UNION
select 
	 YearlyPremium.underwriting_year
	,cl.ClaimType
	,YearlyPremium.IPCoverPremium
	,sum(claimpaidamount) as TotalClaimPaid
	,YearlyPremium.IPCoverPremium - sum(cl.claimpaidamount) as CoverProfit 
from
	[dbo].[MemberClaims] cl

outer apply YearlyPremium

where
	underwriting_year is not null and ClaimType ='IP' or ClaimType ='TIB'
	and year(claimpaiddate) between 2010 and 2014
group by 
	 YearlyPremium.underwriting_year
	,YearlyPremium.IPCoverPremium
	,cl.ClaimType 

/*

*/