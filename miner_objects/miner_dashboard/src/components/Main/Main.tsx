import { MinerData } from "../../types";
import { PowerBiDashboard } from "../PowerBiDashboard";

interface MainProps {
  data: MinerData;
}

export const Main = ({ data }: MainProps) => {
  return <PowerBiDashboard data={data} />;
};
