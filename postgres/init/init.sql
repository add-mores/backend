CREATE TABLE public.disease (
	dis_num int NULL,
	disnm_ko varchar NULL,
	disnm_en varchar NULL,
	category varchar NULL,
	dep varchar NULL,
	organ varchar NULL,
	def varchar NULL,
	coo varchar NULL,
	sym varchar NULL,
	sym_k varchar NULL,
	lapse varchar NULL,
	diag varchar NULL,
	therapy varchar NULL,
	guide varchar NULL,
	pvt varchar NULL
);
COMMENT ON TABLE public.disease IS '병에 대한 정보 테이블';