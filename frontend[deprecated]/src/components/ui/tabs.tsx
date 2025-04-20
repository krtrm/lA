import * as React from "react";
import { cn } from "../../utils/cn";

const Tabs = ({ defaultValue, ...props }) => {
  const [value, setValue] = React.useState(defaultValue);
  
  return (
    <TabsContext.Provider value={{ value, setValue }}>
      <div {...props} />
    </TabsContext.Provider>
  );
};

const TabsContext = React.createContext({
  value: "",
  setValue: (value: string) => {}
});

const TabsList = ({ className, ...props }) => (
  <div
    className={cn(
      "flex flex-wrap gap-2 items-center bg-white/5 rounded-lg p-1",
      className
    )}
    {...props}
  />
);

interface TabsTriggerProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  value: string;
}

const TabsTrigger = ({ className, value, ...props }: TabsTriggerProps) => {
  const { value: selectedValue, setValue } = React.useContext(TabsContext);
  const isSelected = selectedValue === value;
  
  return (
    <button
      className={cn(
        "px-4 py-2 rounded-md text-sm font-medium transition-colors",
        isSelected 
          ? "bg-primary text-white" 
          : "text-white/60 hover:text-white hover:bg-white/10",
        className
      )}
      onClick={() => setValue(value)}
      type="button"
      {...props}
    />
  );
};

const TabsContent = ({ className, value, ...props }) => {
  const { value: selectedValue } = React.useContext(TabsContext);
  const isSelected = selectedValue === value;
  
  if (!isSelected) return null;
  
  return <div className={cn("mt-2", className)} {...props} />;
};

export { Tabs, TabsList, TabsTrigger, TabsContent };